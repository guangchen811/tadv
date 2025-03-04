from abc import ABC, abstractmethod

import oyaml as yaml

from cadv_exploration.error_injection.abstract_corruption import DataCorruption
from error_injection.corrupts import MissingCategoricalValueCorruption, GaussianNoise, Scaling, ColumnInserting, \
    MaskValues, ColumnDropping


class AbstractErrorInjectionManager(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def error_injection(self, corrupts: list[DataCorruption]):
        raise NotImplementedError

    def _create_processed_data_path(self, processed_data_dir):
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_idx = len(list(processed_data_dir.iterdir()))
        processed_data_path = processed_data_dir / f"{processed_data_idx}"
        processed_data_path.mkdir(parents=True, exist_ok=True)
        return processed_data_path

    @property
    def corruption_classes(self):
        return {
            "MissingCategoricalValueCorruption": MissingCategoricalValueCorruption,
            "GaussianNoise": GaussianNoise,
            "Scaling": Scaling,
            "ColumnInserting": ColumnInserting,
            "MaskValues": MaskValues,
            "ColumnDropping": ColumnDropping
        }

    def load_error_injection_config(self, error_injection_config_path):
        with open(error_injection_config_path, "r") as f:
            config_data = yaml.safe_load(f)

        corrupts = []
        for entry in config_data:
            for class_name, attributes in entry.items():
                if class_name in self.corruption_classes:
                    corrupt_class = self.corruption_classes[class_name]
                    corrupt_instance = corrupt_class(columns=attributes["Columns"], **attributes["Params"])
                    corrupts.append(corrupt_instance)
                else:
                    raise ValueError(f"Unknown corruption class: {class_name}")

        return corrupts
