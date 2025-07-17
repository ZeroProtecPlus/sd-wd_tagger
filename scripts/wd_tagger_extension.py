import os
import gradio as gr
from modules import script_callbacks, scripts

from core.predictor import WaifuDiffusionPredictor
from ui.interface import WaifuDiffusionUI
from core.config import WDTaggerConfig

class WaifuDiffusionTagger:
    """Main class for WaifuDiffusion Tagger"""
    
    def __init__(self):
        self.predictor = WaifuDiffusionPredictor()
        self.config = WDTaggerConfig()
        self.ui_manager = WaifuDiffusionUI(self.predictor, self.config)

def on_ui_tabs():
    """Register the extension as a standalone tab"""
    tagger = WaifuDiffusionTagger()
    
    # Crear la interfaz dentro del contexto correcto
    with gr.Blocks(analytics_enabled=False) as wd_tagger_interface:
        # Crear la interfaz del tagger
        tagger.ui_manager.create_interface()
    
    # Retornar la tupla correcta: (interface, title, element_id)
    return [(wd_tagger_interface, "WD Tagger", "wd_tagger_tab")]

def load_css():
    """Load CSS styles for the extension"""
    css_path = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)