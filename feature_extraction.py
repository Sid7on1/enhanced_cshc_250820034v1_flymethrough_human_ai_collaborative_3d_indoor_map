# feature_extraction.py

import logging
import numpy as np
import torch
from torch import nn
from typing import Dict, List, Tuple
from computer_vision.config import Config
from computer_vision.utils import load_model, load_data
from computer_vision.exceptions import FeatureExtractionError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """
    Base class for feature extractors.
    """
    def __init__(self, config: Config):
        super(FeatureExtractor, self).__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        raise NotImplementedError

class PoiFeatureExtractor(FeatureExtractor):
    """
    Poi feature extractor.
    """
    def __init__(self, config: Config):
        super(PoiFeatureExtractor, self).__init__(config)
        self.model = load_model(config.model_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        output = self.model(x)
        return output

class CurrentFeatureExtractor(FeatureExtractor):
    """
    Current feature extractor.
    """
    def __init__(self, config: Config):
        super(CurrentFeatureExtractor, self).__init__(config)
        self.model = load_model(config.model_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        output = self.model(x)
        return output

class MeshFeatureExtractor(FeatureExtractor):
    """
    Mesh feature extractor.
    """
    def __init__(self, config: Config):
        super(MeshFeatureExtractor, self).__init__(config)
        self.model = load_model(config.model_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        output = self.model(x)
        return output

class FeatureExtractorManager:
    """
    Manages feature extractors.
    """
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractors: Dict[str, FeatureExtractor] = {}

    def register_feature_extractor(self, name: str, feature_extractor: FeatureExtractor):
        """
        Registers a feature extractor.
        """
        self.feature_extractors[name] = feature_extractor

    def get_feature_extractor(self, name: str) -> FeatureExtractor:
        """
        Gets a feature extractor by name.
        """
        return self.feature_extractors.get(name)

def extract_features(config: Config, data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extracts features from data.
    """
    feature_extractor_manager = FeatureExtractorManager(config)
    feature_extractor_manager.register_feature_extractor('poi', PoiFeatureExtractor(config))
    feature_extractor_manager.register_feature_extractor('current', CurrentFeatureExtractor(config))
    feature_extractor_manager.register_feature_extractor('mesh', MeshFeatureExtractor(config))

    features: Dict[str, torch.Tensor] = {}
    for name, feature_extractor in feature_extractor_manager.feature_extractors.items():
        try:
            feature = feature_extractor(data)
            features[name] = feature
        except FeatureExtractionError as e:
            logger.error(f"Error extracting feature {name}: {e}")
            raise

    return features

def main():
    config = Config()
    data = load_data(config.data_path)
    features = extract_features(config, data)
    logger.info(f"Extracted features: {features}")

if __name__ == "__main__":
    main()