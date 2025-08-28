"""
Model configuration file for the computer_vision project.

This file contains the configuration settings for the project, including model parameters,
data loading, and logging settings.
"""

import logging
import os
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("config.log"),
        logging.StreamHandler()
    ]
)

class Config:
    """
    Configuration class for the computer_vision project.
    """

    def __init__(self):
        self.model_params: Dict[str, float] = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        self.data_loading_params: Dict[str, str] = {
            "data_path": "/path/to/data",
            "image_size": "224x224"
        }
        self.logging_params: Dict[str, str] = {
            "log_level": "INFO",
            "log_file": "config.log"
        }

    def get_model_params(self) -> Dict[str, float]:
        """
        Get the model parameters.

        Returns:
            Dict[str, float]: Model parameters.
        """
        return self.model_params

    def get_data_loading_params(self) -> Dict[str, str]:
        """
        Get the data loading parameters.

        Returns:
            Dict[str, str]: Data loading parameters.
        """
        return self.data_loading_params

    def get_logging_params(self) -> Dict[str, str]:
        """
        Get the logging parameters.

        Returns:
            Dict[str, str]: Logging parameters.
        """
        return self.logging_params

    def update_model_params(self, params: Dict[str, float]) -> None:
        """
        Update the model parameters.

        Args:
            params (Dict[str, float]): New model parameters.
        """
        self.model_params.update(params)

    def update_data_loading_params(self, params: Dict[str, str]) -> None:
        """
        Update the data loading parameters.

        Args:
            params (Dict[str, str]): New data loading parameters.
        """
        self.data_loading_params.update(params)

    def update_logging_params(self, params: Dict[str, str]) -> None:
        """
        Update the logging parameters.

        Args:
            params (Dict[str, str]): New logging parameters.
        """
        self.logging_params.update(params)


class ModelConfig:
    """
    Model configuration class.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_name: str = "resnet50"
        self.model_weights: str = "resnet50_weights.h5"

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: Model name.
        """
        return self.model_name

    def get_model_weights(self) -> str:
        """
        Get the model weights.

        Returns:
            str: Model weights.
        """
        return self.model_weights


class DataConfig:
    """
    Data configuration class.
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_path: str = config.get_data_loading_params()["data_path"]
        self.image_size: str = config.get_data_loading_params()["image_size"]

    def get_data_path(self) -> str:
        """
        Get the data path.

        Returns:
            str: Data path.
        """
        return self.data_path

    def get_image_size(self) -> str:
        """
        Get the image size.

        Returns:
            str: Image size.
        """
        return self.image_size


class LoggingConfig:
    """
    Logging configuration class.
    """

    def __init__(self, config: Config):
        self.config = config
        self.log_level: str = config.get_logging_params()["log_level"]
        self.log_file: str = config.get_logging_params()["log_file"]

    def get_log_level(self) -> str:
        """
        Get the log level.

        Returns:
            str: Log level.
        """
        return self.log_level

    def get_log_file(self) -> str:
        """
        Get the log file.

        Returns:
            str: Log file.
        """
        return self.log_file


def load_config() -> Config:
    """
    Load the configuration.

    Returns:
        Config: Configuration object.
    """
    config = Config()
    return config


def main():
    config = load_config()
    logging.info("Loaded configuration:")
    logging.info(config.get_model_params())
    logging.info(config.get_data_loading_params())
    logging.info(config.get_logging_params())


if __name__ == "__main__":
    main()