from _bootstrap import ensure_src_on_path, project_root

ensure_src_on_path()

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Run the full DownscaleRainfall pipeline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory to use. Passed to all scripts (and as --output-dir for download).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download step.",
    )
    return parser


def run_step(step_name, script_path, extra_args):
    command = [sys.executable, str(script_path), *extra_args]
    print(f"\n[{step_name}]")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent

    steps = []
    if not args.skip_download:
        download_args = []
        if args.data_dir:
            download_args.extend(["--output-dir", args.data_dir])
        steps.append(("Download data", scripts_dir / "download_data.py", download_args))

    common_args = ["--data-dir", args.data_dir] if args.data_dir else []
    steps.extend(
        [
            ("Data preparation", scripts_dir / "run_data_preparation.py", common_args),
            ("Exploration", scripts_dir / "run_exploration.py", common_args),
            ("Occurrence models", scripts_dir / "run_occurrence_models.py", common_args),
            ("Threshold optimization", scripts_dir / "run_threshold_optimization.py", common_args),
            ("Gamma model", scripts_dir / "run_gamma_model.py", common_args),
            ("Gamma-GLM model", scripts_dir / "run_gamma_glm.py", common_args),
            ("Gamma+GPD extension", scripts_dir / "run_gamma_gpd_extension.py", common_args),
            ("Model comparison", scripts_dir / "run_model_comparison.py", common_args),
        ]
    )

    print(f"Project root: {project_root()}")
    for step_name, script_path, step_args in steps:
        run_step(step_name, script_path, step_args)

    print("\nFull pipeline finished successfully.")


if __name__ == "__main__":
    main()
