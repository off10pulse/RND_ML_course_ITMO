"""Распаковка данных конкурса. По умолчанию нужны только CSV, без фото и JSON."""
import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import tempfile
import zipfile

MARKER = ".petfinder-extraction.json"


def default_data_source():
    data = Path(__file__).resolve().parents[1] / "data"
    archive = data / "petfinder-adoption-prediction.zip"
    return archive if archive.is_file() else data / "raw"


def default_cache_dir():
    return Path(__file__).resolve().parents[1] / "data/raw"


def _target(directory, member):
    name = member.filename
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or PureWindowsPath(name).drive or stat.S_ISLNK(member.external_attr >> 16)):
        raise ValueError(f"Небезопасный путь в ZIP: {name}")
    target = (directory / relative).resolve()
    if directory not in target.parents:
        raise ValueError(f"Путь за пределами папки распаковки: {name}")
    return target


def prepare_archive(source, output=None, include_media=False):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Нет архива: {source}")
    with source.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Вместо ZIP скачан указатель Git LFS. Выполните git lfs pull.")
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Не ZIP-архив: {source}")
    info = source.stat()
    identity = {"source": str(source), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
    # Новая версия архива получает отдельный кэш, не смешиваясь со старой.
    destination = Path(output) if output is not None else (
        default_cache_dir() / source.stem / f"{info.st_size}-{info.st_mtime_ns}"
    )
    destination = destination.resolve()
    if destination == source or destination in source.parents:
        raise ValueError("Папка распаковки не должна содержать исходный ZIP")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / MARKER
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
    if previous is not None and previous.get("archive") != identity:
        raise ValueError("В этой папке распакован другой архив. Укажите другую --output.")
    if previous is None and any(p.name != ".gitkeep" for p in destination.iterdir()):
        raise ValueError("Папка распаковки не пуста. Укажите новую --output.")

    with zipfile.ZipFile(source) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        selected = members if include_media else [m for m in members if m.filename.lower().endswith(".csv")]
        train_files = [PurePosixPath(m.filename) for m in selected
                       if PurePosixPath(m.filename).name == "train.csv"]
        if len(train_files) != 1:
            raise ValueError("В архиве должен быть один train.csv")
        parent = train_files[0].parent
        relative_root = parent.parent if parent.name == "train" else parent
        names = {m.filename for m in selected}
        test_options = [relative_root / "test.csv", relative_root / "test/test.csv"]
        if sum(str(p) in names for p in test_options) != 1:
            raise ValueError("В архиве не найден однозначный test.csv рядом с train")
        targets = [(m, _target(destination, m)) for m in selected]
        if len({p for _, p in targets}) != len(targets):
            raise ValueError("ZIP содержит повторяющиеся пути")
        ready = previous is not None and previous.get("complete") and (
            previous.get("include_media") or not include_media
        )
        if ready and all(p.is_file() and p.stat().st_size == m.file_size for m, p in targets):
            return destination / relative_root
        required = sum(m.file_size for m, p in targets
                       if not p.is_file() or p.stat().st_size != m.file_size)
        if shutil.disk_usage(destination).free < required:
            raise OSError(f"Недостаточно места: нужно ещё около {required / 1024**3:.2f} GiB")
        state = {"archive": identity, "include_media": include_media, "complete": False}
        marker.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Распаковка {'всех файлов' if include_media else 'CSV'}: {len(targets)} файлов", flush=True)
        for number, (member, target) in enumerate(targets, 1):
            if not target.is_file() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".unpack-", delete=False) as stream:
                        temporary = Path(stream.name)
                        with archive.open(member) as contents:
                            shutil.copyfileobj(contents, stream)
                    temporary.replace(target)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
            if include_media and number % 10000 == 0:
                print(f"  {number}/{len(targets)}", flush=True)
        state["complete"] = True
        marker.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination / relative_root


def resolve_data_root(source=None, include_media=False):
    source = Path(source) if source is not None else default_data_source()
    if source.is_file() or source.suffix.lower() == ".zip":
        return prepare_archive(source, include_media=include_media)
    if not source.is_dir():
        raise FileNotFoundError(f"Нет папки данных: {source}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_data_source())
    parser.add_argument("--output", type=Path, help="Пустая папка; по умолчанию кэш в data/raw")
    parser.add_argument("--include-media", action="store_true", help="Также распаковать JPG и JSON")
    args = parser.parse_args()
    print(prepare_archive(args.archive, args.output, args.include_media))


if __name__ == "__main__":
    main()
