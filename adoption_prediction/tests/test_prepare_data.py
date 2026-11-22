import json
from pathlib import Path
import stat
import zipfile

import pytest

from src.data import load_data
from src.prepare_data import MARKER, prepare_archive, resolve_data_root


def make_archive(path, pets, prefix=""):
    test = pets.drop(columns="AdoptionSpeed").copy()
    test.PetID = "test-" + test.PetID
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(prefix + "train/train.csv", pets.to_csv(index=False))
        archive.writestr(prefix + "test/test.csv", test.to_csv(index=False))
        archive.writestr(prefix + "test/sample_submission.csv", test[["PetID"]].to_csv(index=False))
        archive.writestr(prefix + "train_images/pet-0-0-1.jpg", b"image placeholder")
        archive.writestr(prefix + "train_sentiment/pet-0-0.json", "{}")


@pytest.mark.parametrize("prefix", ["", "petfinder/"])
def test_archive_cache_and_optional_media(pets, tmp_path, prefix):
    source = tmp_path / "competition.zip"
    output = tmp_path / "unpacked"
    make_archive(source, pets, prefix)
    root = prepare_archive(source, output)
    train, test = load_data(root)
    assert len(train) == len(test) == len(pets)
    assert not (root / "train_images").exists()
    csv = root / "train/train.csv"
    before = csv.stat().st_mtime_ns
    assert prepare_archive(source, output) == root
    assert csv.stat().st_mtime_ns == before
    csv.unlink()
    assert prepare_archive(source, output) == root
    assert len(load_data(root)[0]) == len(pets)
    assert prepare_archive(source, output, include_media=True) == root
    assert (root / "train_images/pet-0-0-1.jpg").read_bytes() == b"image placeholder"
    assert json.loads((output / MARKER).read_text())["complete"]
    assert json.loads((output / MARKER).read_text())["include_media"]


def test_loading_zip_and_flat_csv_layout(pets, tmp_path, monkeypatch):
    monkeypatch.setattr("src.prepare_data.default_cache_dir", lambda: tmp_path / "cache")
    source = tmp_path / "flat.zip"
    test = pets.drop(columns="AdoptionSpeed").copy()
    test.PetID = "test-" + test.PetID
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("train.csv", pets.to_csv(index=False))
        archive.writestr("test.csv", test.to_csv(index=False))
    train, loaded_test = load_data(source)
    assert len(train) == len(loaded_test) == len(pets)
    assert resolve_data_root(resolve_data_root(source)) == resolve_data_root(source)


@pytest.mark.parametrize("name", ["../outside.csv", "/outside.csv", "C:/outside.csv", "..\\outside.csv"])
def test_unsafe_paths_are_rejected(pets, tmp_path, name):
    source = tmp_path / "unsafe.zip"
    make_archive(source, pets)
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr(name, "unsafe")
    with pytest.raises(ValueError, match="ZIP|пределами"):
        prepare_archive(source, tmp_path / "data")
    assert not (tmp_path / "outside.csv").exists()


def test_symlinks_and_unmanaged_outputs_are_rejected(pets, tmp_path):
    source = tmp_path / "symlink.zip"
    make_archive(source, pets)
    with zipfile.ZipFile(source, "a") as archive:
        link = zipfile.ZipInfo("link.csv")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../outside")
    with pytest.raises(ValueError, match="ZIP"):
        prepare_archive(source, tmp_path / "unpacked")
    output = tmp_path / "my-files"
    output.mkdir()
    (output / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="не пуста"):
        prepare_archive(source, output)
    assert (output / "notes.txt").read_text() == "keep me"


def test_lfs_pointer_has_actionable_error(tmp_path):
    pointer = tmp_path / "competition.zip"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 100\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        prepare_archive(pointer)


def test_changed_archive_gets_a_new_cache(pets, tmp_path, monkeypatch):
    monkeypatch.setattr("src.prepare_data.default_cache_dir", lambda: tmp_path / "cache")
    source = tmp_path / "competition.zip"
    make_archive(source, pets)
    old_root = prepare_archive(source)
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("extra.csv", "a,b\n1,2\n")
    new_root = prepare_archive(source)
    assert new_root != old_root
    assert (old_root / "train/train.csv").is_file()
    assert (new_root / "extra.csv").is_file()
    with pytest.raises(ValueError, match="другой архив"):
        prepare_archive(source, old_root)
