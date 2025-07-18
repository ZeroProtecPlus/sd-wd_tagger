import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    repo_id: str
    display_name: str
    description: str
    version: str

@dataclass
class ThresholdConfig:
    """Configuration for tag thresholds"""
    general_default: float = 0.35
    character_default: float = 0.85
    rating_default: float = 0.5
    slider_step: float = 0.05
    min_character_mcut: float = 0.15

class WDTaggerConfig:
    """Main configuration class for WaifuDiffusion Tagger"""
    
    def __init__(self):
        self.title = "WaifuDiffusion Tagger"
        self.description = """
        Advanced AI-powered image tagging system for anime/manga style images.
        
        Upload an image and get detailed tags for characters, general content, and ratings.
        Perfect for organizing your image collections and generating prompts for AI art.
        """
        
        self.models = self._init_models()
        self.thresholds = ThresholdConfig()
        self.file_config = self._init_file_config()
        self.kaomojis = self._init_kaomojis()
    
    def _init_models(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations"""
        return {
            "swinv2_v3": ModelConfig(
                repo_id="SmilingWolf/wd-swinv2-tagger-v3",
                display_name="SwinV2 v3 (Recommended)",
                description="Latest SwinV2 model with improved accuracy",
                version="v3"
            ),
            "convnext_v3": ModelConfig(
                repo_id="SmilingWolf/wd-convnext-tagger-v3",
                display_name="ConvNext v3",
                description="ConvNext architecture with high performance",
                version="v3"
            ),
            "vit_v3": ModelConfig(
                repo_id="SmilingWolf/wd-vit-tagger-v3",
                display_name="ViT v3",
                description="Vision Transformer model",
                version="v3"
            ),
            "moat_v2": ModelConfig(
                repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2",
                display_name="MOAT v2",
                description="MOAT architecture model",
                version="v2"
            ),
            "swin_v2": ModelConfig(
                repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                display_name="SwinV2 v2",
                description="Previous generation SwinV2 model",
                version="v2"
            ),
            "convnext_v2": ModelConfig(
                repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
                display_name="ConvNext v2",
                description="Previous generation ConvNext model",
                version="v2"
            ),
            "convnextv2_v2": ModelConfig(
                repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                display_name="ConvNextV2 v2",
                description="ConvNextV2 architecture model",
                version="v2"
            ),
            "vit_v2": ModelConfig(
                repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2",
                display_name="ViT v2",
                description="Previous generation Vision Transformer",
                version="v2"
            ),
            "eva02": ModelConfig(
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
            display_name="Eva 02",
            description="Previous generation Vision Transformer",
            version="v2"
            )
        }
    
    def _init_file_config(self) -> Dict[str, str]:
        """Initialize file configuration"""
        return {
            "model_filename": "model.onnx",
            "label_filename": "selected_tags.csv"
        }
    
    def _init_kaomojis(self) -> List[str]:
        """Initialize kaomoji list for tag processing"""
        return [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
            "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o",
            "u_u", "x_x", "|_|", "||_||"
        ]
    
    def get_model_choices(self) -> List[str]:
        """Get list of model choices for dropdown"""
        return [model.repo_id for model in self.models.values()]
    
    def get_model_labels(self) -> List[str]:
        """Get list of model display names"""
        return [model.display_name for model in self.models.values()]
    
    def get_default_model(self) -> str:
        """Get default model repository ID"""
        return self.models["swinv2_v3"].repo_id
    
    def get_hf_token(self) -> str:
        """Get HuggingFace token from environment"""
        return os.environ.get("HF_TOKEN", "")
    
    @property
    def css_classes(self) -> Dict[str, str]:
        """CSS class names for styling"""
        return {
            "main_container": "wd-tagger-container",
            "input_panel": "wd-tagger-input-panel",
            "output_panel": "wd-tagger-output-panel",
            "threshold_row": "wd-tagger-threshold-row",
            "button_row": "wd-tagger-button-row",
            "model_selector": "wd-tagger-model-selector",
            "image_upload": "wd-tagger-image-upload",
            "tag_output": "wd-tagger-tag-output",
            "rating_output": "wd-tagger-rating-output",
            "character_output": "wd-tagger-character-output"
        }