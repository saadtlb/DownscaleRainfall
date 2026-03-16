import argparse
from pathlib import Path
import sys

REQUIRED_DATA_FILES = [
    "station.data.81.10.txt",
    "ERA5.slp.81.10.txt",
    "ERA5.d2.81.10.txt",
    "ERA5.lon.81.10.txt",
    "ERA5.lat.81.10.txt",
]


def project_root():
    return Path(__file__).resolve().parents[1]


def source_dir():
    return project_root() / "src"


def local_data_dir():
    return project_root() / "data" / "raw"


def has_required_data(data_dir):
    data_dir = Path(data_dir)
    return all((data_dir / filename).exists() for filename in REQUIRED_DATA_FILES)


def default_data_dir():
    local_dir = local_data_dir()
    if local_dir.exists() and has_required_data(local_dir):
        return local_dir
    return project_root().parents[0] / "Data"


def resolve_data_dir(data_dir=None):
    if data_dir is not None:
        return Path(data_dir).resolve()
    return default_data_dir()


def add_data_dir_argument(parser):
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Chemin local vers les donnees. Par defaut: data/raw si present, sinon ../Data",
    )
    return parser


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    add_data_dir_argument(parser)
    return parser


def ensure_src_on_path():
    src = str(source_dir())
    if src not in sys.path:
        sys.path.insert(0, src)
