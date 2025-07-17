import gradio as gr
from typing import Dict, Any, Tuple, List
from core.predictor import WaifuDiffusionPredictor
from core.config import WDTaggerConfig
import json

class WaifuDiffusionUI:
    """Main UI class for WaifuDiffusion Tagger"""
    
    def __init__(self, predictor: WaifuDiffusionPredictor, config: WDTaggerConfig):
        self.predictor = predictor
        self.config = config
        self.components = {}
    
    def create_interface(self) -> Dict[str, Any]:
        """Create the main interface components"""
        with gr.Row(elem_classes=[self.config.css_classes["main_container"]]):
            with gr.Column(variant="panel", elem_classes=[self.config.css_classes["input_panel"]]):
                # Header
                gr.HTML(
                    f"""
                    <script src="file=E:/SD_Matrix/Packages/reforge/extensions/sd-wd_tagger/ui/clipboard.js"></script>
                    <div class='wd-tagger-header'>
                        <h2>{self.config.title}</h2>
                        <p>{self.config.description}</p>
                    </div>
                    """
                )
                
                # Image upload
                image_input = gr.Image(
                    type="pil",
                    image_mode="RGBA",
                    label="Upload Image",
                    elem_classes=[self.config.css_classes["image_upload"]]
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=self.config.get_model_choices(),
                    value=self.config.get_default_model(),
                    label="Model Selection",
                    info="Choose the AI model for tagging",
                    elem_classes=[self.config.css_classes["model_selector"]]
                )
                
                # Advanced settings
                with gr.Accordion("Advanced Settings", open=True):
                    with gr.Row(elem_classes=[self.config.css_classes["threshold_row"]]):
                        general_thresh = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=self.config.thresholds.slider_step,
                            value=self.config.thresholds.general_default,
                            label="General Tags Threshold",
                            info="Lower = more tags, Higher = fewer but more confident tags",
                            elem_classes=["wd-tagger-slider"]
                        )
                        general_mcut = gr.Checkbox(
                            value=False,
                            label="Auto MCut",
                            info="Automatic threshold detection",
                            elem_classes=["wd-tagger-checkbox"]
                        )
                    
                    with gr.Row(elem_classes=[self.config.css_classes["threshold_row"]]):
                        character_thresh = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=self.config.thresholds.slider_step,
                            value=self.config.thresholds.character_default,
                            label="Character Tags Threshold",
                            info="Threshold for character detection",
                            elem_classes=["wd-tagger-slider"]
                        )
                        character_mcut = gr.Checkbox(
                            value=False,
                            label="Auto MCut",
                            info="Automatic threshold detection",
                            elem_classes=["wd-tagger-checkbox"]
                        )
                    
                    # Batch processing
                    with gr.Row():
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=1,
                            label="Batch Size",
                            info="Process multiple images at once",
                            visible=False  # Hidden for now
                        )
                
                # Action buttons
                with gr.Row(elem_classes=[self.config.css_classes["button_row"]]):
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        elem_classes=["wd-tagger-button", "wd-tagger-clear-button"]
                    )
                    predict_btn = gr.Button(
                        "Generate Tags",
                        variant="primary",
                        size="lg",
                        elem_classes=["wd-tagger-button", "wd-tagger-predict-button"]
                    )
            
            # Output panel
            with gr.Column(variant="panel", elem_classes=[self.config.css_classes["output_panel"]]):
                # Main outputs
                with gr.Tabs(elem_classes=["wd-tagger-output-tabs"]):
                    with gr.TabItem("🏷️ Standard Tags", elem_classes=["wd-tagger-tab"]):
                        standard_output = gr.Textbox(
                            label="Formatted Tags",
                            placeholder="Tags will appear here...",
                            lines=4,
                            max_lines=8,
                            elem_classes=[self.config.css_classes["tag_output"]]
                        )
                        
                        copy_standard_btn = gr.Button(
                            "📋 Copy to Clipboard",
                            size="sm",
                            elem_classes=["wd-tagger-copy-button"]
                        )
                    
                    with gr.TabItem("🔞 R34 Format", elem_classes=["wd-tagger-tab"]):
                        r34_output = gr.Textbox(
                            label="R34 Compatible Tags",
                            placeholder="R34 formatted tags will appear here...",
                            lines=4,
                            max_lines=8,
                            elem_classes=[self.config.css_classes["tag_output"]]
                        )
                        
                        copy_r34_btn = gr.Button(
                            "📋 Copy to Clipboard",
                            size="sm",
                            elem_classes=["wd-tagger-copy-button"]
                        )
                    
                    with gr.TabItem("⭐ Ratings", elem_classes=["wd-tagger-tab"]):
                        rating_output = gr.JSON(
                            label="Content Ratings",
                            elem_classes=[self.config.css_classes["rating_output"]]
                        )
                    
                    with gr.TabItem("👥 Characters", elem_classes=["wd-tagger-tab"]):
                        character_output = gr.JSON(
                            label="Detected Characters",
                            elem_classes=[self.config.css_classes["character_output"]]
                        )
                    
                    with gr.TabItem("🔍 All Tags", elem_classes=["wd-tagger-tab"]):
                        all_tags_output = gr.JSON(
                            label="All General Tags with Confidence",
                            elem_classes=["wd-tagger-all-tags"]
                        )
                
                # Additional info
                with gr.Accordion("Processing Info", open=True):
                    processing_info = gr.Markdown(
                        "Processing information will appear here after tagging.",
                        elem_classes=["wd-tagger-info"]
                    )
        
        # Store components for event handling
        self.components = {
            "image_input": image_input,
            "model_dropdown": model_dropdown,
            "general_thresh": general_thresh,
            "general_mcut": general_mcut,
            "character_thresh": character_thresh,
            "character_mcut": character_mcut,
            "predict_btn": predict_btn,
            "clear_btn": clear_btn,
            "standard_output": standard_output,
            "r34_output": r34_output,
            "rating_output": rating_output,
            "character_output": character_output,
            "all_tags_output": all_tags_output,
            "processing_info": processing_info,
            "copy_standard_btn": copy_standard_btn,
            "copy_r34_btn": copy_r34_btn
        }
        
        # Set up event handlers
        self._setup_event_handlers()
        
        return self.components
    
    def create_tab_interface(self):
        """Create the tab interface for the extension"""
        self.create_interface()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the interface"""
        # Main prediction
        self.components["predict_btn"].click(
            fn=self._predict_wrapper,
            inputs=[
                self.components["image_input"],
                self.components["model_dropdown"],
                self.components["general_thresh"],
                self.components["general_mcut"],
                self.components["character_thresh"],
                self.components["character_mcut"]
            ],
            outputs=[
                self.components["standard_output"],
                self.components["r34_output"],
                self.components["rating_output"],
                self.components["character_output"],
                self.components["all_tags_output"],
                self.components["processing_info"]
            ]
        )
        
        # Clear button
        self.components["clear_btn"].click(
            fn=self._clear_all,
            outputs=[
                self.components["image_input"],
                self.components["standard_output"],
                self.components["r34_output"],
                self.components["rating_output"],
                self.components["character_output"],
                self.components["all_tags_output"],
                self.components["processing_info"]
            ]
        )
        
        # Copy buttons with JS
        self.components["copy_standard_btn"].click(
            fn=None,
            inputs=[self.components["standard_output"]],
            outputs=[],
            _js="copyToClipboard"
        )
        
        self.components["copy_r34_btn"].click(
            fn=None,
            inputs=[self.components["r34_output"]],
            outputs=[],
            _js="copyToClipboard"
        )
    
    def _predict_wrapper(self, image, model_repo, general_thresh, general_mcut, character_thresh, character_mcut):
        """Wrapper for the prediction function with UI updates"""
        if image is None:
            return (
                "",
                "",
                {},
                {},
                {},
                "No image provided for processing."
            )
        
        try:
            # Run prediction
            standard_tags, r34_tags, rating_dict, character_dict, general_dict = self.predictor.predict(
                image, model_repo, general_thresh, general_mcut, character_thresh, character_mcut
            )
            
            # Create processing info
            processing_info = self._create_processing_info(
                model_repo, general_thresh, character_thresh, 
                len(general_dict), len(character_dict)
            )
            
            return (
                standard_tags,
                r34_tags,
                rating_dict,
                character_dict,
                general_dict,
                processing_info
            )
            
        except Exception as e:
            error_info = f"**Error Details:**\n```\n{str(e)}\n```"
            
            return (
                "",
                "",
                {},
                {},
                {},
                error_info
            )
    
    def _clear_all(self):
        """Clear all outputs"""
        return (
            None,  # image
            "",    # standard_output
            "",    # r34_output
            {},
            {},
            {},
            "Upload an image to begin tagging."
        )
    
    def _create_processing_info(self, model_repo, general_thresh, character_thresh, general_count, character_count):
        """Create processing information display"""
        model_name = model_repo.split('/')[-1] if '/' in model_repo else model_repo
        
        return f"""
        ### Processing Details
        
        **Model Used:** `{model_name}`
        
        **Thresholds:**
        - General Tags: `{general_thresh:.2f}`
        - Character Tags: `{character_thresh:.2f}`
        
        **Results:**
        - General Tags Found: `{general_count}`
        - Character Tags Found: `{character_count}`
        
        **Model Repository:** `{model_repo}`
        """
    
    def get_example_inputs(self):
        """Get example inputs for the interface"""
        return [
            ["example1.jpg", self.config.get_default_model(), 0.35, False, 0.85, False],
            ["example2.jpg", self.config.get_default_model(), 0.25, True, 0.75, True],
        ]
    
def get_gradio_interface(predictor: WaifuDiffusionPredictor, config: WDTaggerConfig):
    """Get the Gradio interface for the tagger"""
    ui = WaifuDiffusionUI(predictor, config)
    interface = ui.create_interface()
    
    return [(interface, "WD Tagger", "wd_tagger")]
