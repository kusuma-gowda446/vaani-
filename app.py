#!/usr/bin/env python3
"""
Vaani ASR - Simple Testing UI
"""

import os
import subprocess
import tempfile
import gradio as gr
import torch
from transformers import AutoModel
from dotenv import load_dotenv

# Load token
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')

REPO = 'ARTPARK-IISc/SraVaani-1.0'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model once
print(f'🔄 Loading model on {DEV}...')
if TOKEN:
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
else:
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
print('✅ Model loaded!')

def transcribe(audio_path):
    """Simple transcription function"""
    if audio_path is None:
        return "Please upload an audio file"
    
    try:
        # Convert if not WAV
        if not audio_path.lower().endswith('.wav'):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000',
                temp_wav, '-y', '-loglevel', 'error'
            ], check=True)
            audio_path = temp_wav
        
        # Transcribe
        result = model.transcribe([audio_path])[0]
        
        # Clean up
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create simple UI
with gr.Blocks(title="Vaani ASR Test") as demo:
    gr.Markdown("# 🎙️ Vaani ASR - Testing UI")
    gr.Markdown("Upload audio or record from microphone")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath",
                label="Upload or Record Audio"
            )
            transcribe_btn = gr.Button("🔊 Transcribe", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="📝 Transcription",
                lines=4,
                placeholder="Result will appear here..."
            )
    
    # Connect
    transcribe_btn.click(transcribe, inputs=audio_input, outputs=output)
    audio_input.change(transcribe, inputs=audio_input, outputs=output)
    
    gr.Markdown("---")
    gr.Markdown("✅ Supports: WAV, MP3, M4A, FLAC")

if __name__ == "__main__":
    print("\n🚀 Starting UI...")
    print("🌐 Open: http://127.0.0.1:7860\n")
    demo.launch(server_name="127.0.0.1", server_port=7860)
