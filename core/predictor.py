import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import huggingface_hub
import onnxruntime as rt

from core.config import WDTaggerConfig
from core.tag_processor import TagProcessor

class WaifuDiffusionPredictor:
    """Main predictor class for WaifuDiffusion Tagger"""
    
    def __init__(self):
        self.config = WDTaggerConfig()
        self.tag_processor = TagProcessor(self.config)
        self.model = None
        self.model_target_size = None
        self.last_loaded_repo = None
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
    
    def download_model(self, model_repo: str) -> Tuple[str, str]:
        """Download model files from HuggingFace Hub"""
        try:
            csv_path = huggingface_hub.hf_hub_download(
                model_repo,
                self.config.file_config["label_filename"],
                token=self.config.get_hf_token() or None,
            )
            model_path = huggingface_hub.hf_hub_download(
                model_repo,
                self.config.file_config["model_filename"],
                token=self.config.get_hf_token() or None,
            )
            return csv_path, model_path
        except Exception as e:
            raise Exception(f"Failed to download model from {model_repo}: {str(e)}")
    
    def load_labels(self, dataframe: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[int]]:
        """Load and process labels from CSV file"""
        name_series = dataframe["name"]
        name_series = name_series.map(
            lambda x: x.replace("_", " ") if x not in self.config.kaomojis else x
        )
        tag_names = name_series.tolist()
        
        rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        general_indexes = list(np.where(dataframe["category"] == 0)[0])
        character_indexes = list(np.where(dataframe["category"] == 4)[0])
        
        return tag_names, rating_indexes, general_indexes, character_indexes
    
    def load_model(self, model_repo: str) -> bool:
        """Load model and labels"""
        if model_repo == self.last_loaded_repo:
            return True
        
        try:
            csv_path, model_path = self.download_model(model_repo)
            
            # Load labels
            tags_df = pd.read_csv(csv_path)
            sep_tags = self.load_labels(tags_df)
            
            self.tag_names = sep_tags[0]
            self.rating_indexes = sep_tags[1]
            self.general_indexes = sep_tags[2]
            self.character_indexes = sep_tags[3]
            
            # Load model
            self.model = rt.InferenceSession(model_path)
            _, height, width, _ = self.model.get_inputs()[0].shape
            self.model_target_size = height
            
            self.last_loaded_repo = model_repo
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def prepare_image(self, image: Image.Image) -> np.ndarray:
        """Prepare image for model input"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        target_size = self.model_target_size
        
        # Create white background and composite
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
        
        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2
        
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        # Resize to target size
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.LANCZOS
            )
        
        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        
        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]
        
        return np.expand_dims(image_array, axis=0)
    
    def mcut_threshold(self, probs: np.ndarray) -> float:
        """
        Maximum Cut Thresholding (MCut)
        Largeron, C., Moulin, C., & Gery, M. (2012)
        """
        if len(probs) == 0:
            return 0.0
        
        sorted_probs = probs[probs.argsort()[::-1]]
        if len(sorted_probs) <= 1:
            return sorted_probs[0] if len(sorted_probs) == 1 else 0.0
        
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return thresh
    
    def predict(
        self,
        image: Image.Image,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        rating_thresh: float = 0.5
    ) -> Tuple[str, str, Dict, Dict, Dict]:
        """
        Main prediction function
        Returns: (formatted_tags, r34_tags, rating_dict, character_dict, general_dict)
        """
        if not self.load_model(model_repo):
            return "Model loading failed", "", {}, {}, {}
        
        if image is None:
            return "No image provided", "", {}, {}, {}
        
        try:
            # Prepare image
            processed_image = self.prepare_image(image)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            preds = self.model.run([label_name], {input_name: processed_image})[0]
            
            # Process predictions
            labels = list(zip(self.tag_names, preds[0].astype(float)))
            
            # Process ratings
            rating_labels = [labels[i] for i in self.rating_indexes]
            rating_dict = dict(rating_labels)
            
            # Process general tags
            general_labels = [labels[i] for i in self.general_indexes]
            
            if general_mcut_enabled:
                general_probs = np.array([x[1] for x in general_labels])
                general_thresh = self.mcut_threshold(general_probs)
            
            general_results = [x for x in general_labels if x[1] > general_thresh]
            general_dict = dict(general_results)
            
            # Process character tags
            character_labels = [labels[i] for i in self.character_indexes]
            
            if character_mcut_enabled:
                character_probs = np.array([x[1] for x in character_labels])
                character_thresh = self.mcut_threshold(character_probs)
                character_thresh = max(self.config.thresholds.min_character_mcut, character_thresh)
            
            character_results = [x for x in character_labels if x[1] > character_thresh]
            character_dict = dict(character_results)
            
            # Format tags
            formatted_tags = self.tag_processor.format_standard_tags(general_results)
            r34_tags = self.tag_processor.format_r34_tags(general_results)
            
            return formatted_tags, r34_tags, rating_dict, character_dict, general_dict
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            return error_msg, "", {}, {}, {}
    
    def batch_predict(
        self,
        images: List[Image.Image],
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool
    ) -> List[Tuple]:
        """Batch prediction for multiple images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.predict(
                    image, model_repo, general_thresh, general_mcut_enabled,
                    character_thresh, character_mcut_enabled
                )
                results.append(result)
                print(f"Processed image {i+1}/{len(images)}")
            except Exception as e:
                error_result = (f"Error processing image {i+1}: {str(e)}", "", {}, {}, {})
                results.append(error_result)
        
        return results