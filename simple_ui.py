#!/usr/bin/env python3
"""
Pragna Vaani - Complete Kannada Speech Suite
1. Kannada Speech → Kannada Text (ASR)
2. English Speech → English Text → Kannada Text (Translation)
"""

import os
import subprocess
import tempfile
import time
import gradio as gr
import torch
from dotenv import load_dotenv

# Force CPU mode
torch.cuda.is_available = lambda: False
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load token
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')

# ============================================================
# PART 1: Kannada ASR Model (SraVaani-1.0)
# ============================================================
REPO = 'ARTPARK-IISc/SraVaani-1.0'
DEV = 'cpu'  # Force CPU

def add_kannada_punctuation(text):
    """Add simple punctuation to Kannada text"""
    text = ' '.join(text.split())
    if text and not text[-1] in ['.', '।', '?', '!']:
        text = text + '।'
    if text.endswith('.'):
        text = text[:-1] + '।'
    return text

print('🔄 Loading Kannada ASR model (SraVaani-1.0)...')
from transformers import AutoModel
if TOKEN:
    asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
else:
    asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
print('✅ Kannada ASR model ready!\n')

# ============================================================
# PART 2: English-to-Kannada Translation Models
# ============================================================
print('🔄 Loading English ASR model (Whisper)...')
import whisper
whisper_model = whisper.load_model("base")
print('✅ Whisper model ready!')

print('🔄 Loading English-to-Kannada Translation model (NLLB)...')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
trans_model_name = "facebook/nllb-200-distilled-600M"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(DEV).eval()
print('✅ Translation model ready!\n')

# ============================================================
# Core Functions
# ============================================================

def convert_audio(audio_path):
    """Convert audio to WAV format (16kHz, mono)"""
    if audio_path is None:
        return None
    
    if not audio_path.lower().endswith('.wav'):
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        subprocess.run([
            'ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000',
            temp_wav, '-y', '-loglevel', 'error'
        ], check=True)
        return temp_wav
    return audio_path

def cleanup_temp_file(file_path):
    """Clean up temporary file"""
    if file_path and os.path.exists(file_path) and 'temp' in file_path:
        os.remove(file_path)

def transcribe_kannada(audio_path):
    """Kannada ASR: Speech → Kannada Text"""
    if audio_path is None:
        return "Please upload or record Kannada audio", "", ""
    
    temp_file = None
    try:
        temp_file = convert_audio(audio_path)
        if temp_file:
            audio_path = temp_file
        
        # Transcribe
        raw_result = asr_model.transcribe([audio_path])[0]
        punctuated = add_kannada_punctuation(raw_result)
        
        cleanup_temp_file(temp_file)
        return punctuated, "", ""
    
    except Exception as e:
        cleanup_temp_file(temp_file)
        return "", "", f"Error: {e}"

def translate_english_to_kannada(audio_path):
    """English Speech → English Text → Kannada Text"""
    if audio_path is None:
        return "Please upload or record English audio", "", ""
    
    temp_file = None
    try:
        temp_file = convert_audio(audio_path)
        if temp_file:
            audio_path = temp_file
        
        # Step 1: Transcribe English speech to English text
        result = whisper_model.transcribe(audio_path, language="en")
        english_text = result["text"]
        
        # Step 2: Translate English text to Kannada text
        inputs = trans_tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = translation_model.generate(
                **inputs,
                forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids("kan_Knda"),  # ✅ FIXED
                max_length=200,
                num_beams=5,
                early_stopping=True
            )
        kannada_text = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        cleanup_temp_file(temp_file)
        return english_text, kannada_text, ""
    
    except Exception as e:
        cleanup_temp_file(temp_file)
        return "", "", f"Error: {e}"

def process(audio, mode):
    """Route to appropriate function based on mode"""
    if mode == "Kannada ASR":
        result, _, error = transcribe_kannada(audio)
        return result, "", error
    else:
        english, kannada, error = translate_english_to_kannada(audio)
        return kannada, english, error

# ============================================================
# UI: Combined Interface
# ============================================================

with gr.Blocks(title="Pragna Vaani - Complete") as demo:
    gr.Markdown("""
    # 🎙️ Pragna Vaani - Complete Speech Suite
    
    | Mode | Input | Output |
    |------|-------|--------|
    | **Kannada ASR** | Kannada Speech | Kannada Text |
    | **English → Kannada** | English Speech | Kannada Text |
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Kannada ASR", "English → Kannada"],
                label="Select Mode",
                value="Kannada ASR"
            )
            
            audio = gr.Audio(
                type="filepath",
                label="Upload or Record Audio"
            )
            
            btn = gr.Button("Transcribe", variant="primary")
            clear_btn = gr.Button("Clear All", variant="secondary")
        
        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="Transcription Result",
                lines=6,
                placeholder="Result will appear here..."
            )
            
            intermediate_box = gr.Textbox(
                label="Intermediate (English Text)",
                lines=2,
                placeholder="For English → Kannada mode, shows transcribed English...",
                interactive=False
            )
            
            error_box = gr.Textbox(
                label="Errors",
                lines=2,
                placeholder="Any errors will appear here...",
                interactive=False
            )
    
    # Connect buttons
    btn.click(
        process,
        inputs=[audio, mode],
        outputs=[output_box, intermediate_box, error_box]
    )
    
    audio.change(
        process,
        inputs=[audio, mode],
        outputs=[output_box, intermediate_box, error_box]
    )
    
    mode.change(
        process,
        inputs=[audio, mode],
        outputs=[output_box, intermediate_box, error_box]
    )
    
    clear_btn.click(
        lambda: (None, "", "", ""),
        outputs=[audio, output_box, intermediate_box, error_box]
    )
    
    gr.Markdown("""
    ---
    ### 📌 How to Use
    
    1. **Select Mode**: Kannada ASR or English → Kannada
    2. **Upload Audio**: Upload file or record
    3. **Click Transcribe**: Get your result
    """)

# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Pragna Vaani - Complete Speech Suite")
    print("🌐 Open: http://127.0.0.1:7860")
    print("📝 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
