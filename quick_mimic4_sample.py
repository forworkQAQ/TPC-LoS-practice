"""Generate a small MIMIC-IV sample dataset for quick experiments."""

import json
from MIMIC_preprocessing.timeseries import timeseries_main
from MIMIC_preprocessing.flat_and_labels import flat_and_labels_main
from eICU_preprocessing.split_train_test import split_train_test


def main() -> None:
    with open('paths.json', 'r') as f:
        path = json.load(f)["MIMIC_path"]
    timeseries_main(path, test=True)
    flat_and_labels_main(path)
    split_train_test(path, is_test=True, MIMIC=True)


if __name__ == "__main__":
    main()

