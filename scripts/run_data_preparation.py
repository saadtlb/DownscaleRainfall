from _bootstrap import build_parser, ensure_src_on_path, resolve_data_dir

ensure_src_on_path()

from downscale_precipitation.data.loading import load_prepared_data
from downscale_precipitation.data.temporal_masks import build_train_test_masks


def main():
    parser = build_parser("Charge les donnees et affiche les dimensions principales.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data = load_prepared_data(data_dir)
    mask_train, mask_test = build_train_test_masks()

    print("Data loaded successfully.")
    print(f"Data directory             : {data_dir}")
    print(f"Stations table shape       : {data['stations'].shape}")
    print(f"Stations data shape        : {data['stations_data'].shape}")
    print(f"SLP shape                  : {data['slp'].shape}")
    print(f"d2 shape                   : {data['d2'].shape}")
    print(f"Train winter days          : {int(mask_train.sum())}")
    print(f"Test winter days           : {int(mask_test.sum())}")


if __name__ == "__main__":
    main()
