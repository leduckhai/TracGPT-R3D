from typing import List, Optional, Tuple, Union, Any
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()
ROOT = os.getenv("ROOT")
import sys

sys.path.append(ROOT)
import torch.nn.functional as F
from transformers import Phi3Config
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union, Any, Dict
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json


@dataclass
class MultimodalConfig:
    """Configuration for multimodal components"""

    vision_tower: Optional[str] = "vit3d"
    bbox3d_token_id: Optional[int] = None
    
    # bbox config
    img_token_id: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MultimodalConfig":
        """Create from dictionary"""
        return cls(**config_dict)


class TracPhi3Config(Phi3Config):
    """Configuration for Trac Phi3 model"""

    model_type = "trac-phi3"

    def __init__(self, **kwargs):
        self.multimodal_keys = [
            "vision_tower",
            "bbox3d_token_id",
            # Projector specific

            "img_token_id",
        ]

        # Extract and store multimodal parameters as individual attributes
        # with default values from MultimodalConfig
        default_multimodal = MultimodalConfig()

        for key in self.multimodal_keys:
            default_value = getattr(default_multimodal, key)
            value = kwargs.pop(
                key, default_value
            )  # Remove from kwargs to avoid conflicts
            setattr(self, key, value)

        # Initialize parent config (kwargs now clean of multimodal params)
        super().__init__(**kwargs)
        

        # DO NOT store MultimodalConfig object - it causes JSON serialization issues
        # Instead, we'll reconstruct it on-demand via the multimodal property

    @property
    def multimodal(self) -> MultimodalConfig:
        """Get multimodal config object (reconstructed on-demand)"""
        multimodal_dict = {}
        for key in self.multimodal_keys:
            if hasattr(self, key):
                multimodal_dict[key] = getattr(self, key)
        return MultimodalConfig.from_dict(multimodal_dict)

    def to_dict(self) -> Dict:
        """Override to ensure all attributes are JSON serializable"""
        output = super().to_dict()

        # Add multimodal attributes to the output dict
        for key in self.multimodal_keys:
            if hasattr(self, key):
                output[key] = getattr(self, key)

        # Remove any non-serializable objects that might have been added
        keys_to_remove = []
        for k, v in output.items():
            try:
                json.dumps(v)  # Test if value is JSON serializable
            except (TypeError, ValueError):
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del output[k]

        return output

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "TracPhi3Config":
        """Create config from dictionary"""
        # Merge config_dict and kwargs
        merged_config = {**config_dict, **kwargs}
        return cls(**merged_config)

    def __repr__(self):
        """Custom repr to avoid JSON serialization issues"""
        try:
            return super().__repr__()
        except (TypeError, ValueError) as e:
            # Fallback representation if JSON serialization fails
            return f"{self.__class__.__name__}(model_type='{self.model_type}', hidden_size={getattr(self, 'hidden_size', 'unknown')})"
