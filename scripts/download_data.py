from pathlib import Path

from _bootstrap import ensure_src_on_path, local_data_dir

ensure_src_on_path()

import argparse

from huggingface_hub import hf_hub_download


DATA_FILES = [
    "station.data.81.10.txt",
    "ERA5.slp.81.10.txt",
    "ERA5.d2.81.10.txt",
    "ERA5.lon.81.10.txt",
    "ERA5.lat.81.10.txt",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Telecharge les donnees du projet depuis Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default="saadtaleb/precipitations-era5-stations",
        help="Identifiant du dataset Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(local_data_dir()),
        help="Dossier local de destination pour les donnees.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset Hugging Face : {args.repo_id}")
    print(f"Output directory     : {output_dir}")

    for filename in DATA_FILES:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded: {local_path}")


if __name__ == "__main__":
    main()
