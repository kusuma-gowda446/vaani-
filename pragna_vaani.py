#!/usr/bin/env python3
"""
Pragna Vaani (ಪ್ರಜ್ಞಾ ವಾಣಿ) - Kannada Speech Recognition UI
Voice of Wisdom - Powered by SraVaani-1.0
"""

import os
import subprocess
import tempfile
import gradio as gr
import torch
from transformers import AutoModel
from dotenv import load_dotenv

# Load token from .env file (safe - not in git)
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')

REPO = 'ARTPARK-IISc/SraVaani-1.0'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

print('='*60)
print('🎙️  Pragna Vaani (ಪ್ರಜ್ಞಾ ವಾಣಿ)')
print('   Voice of Wisdom - Kannada Speech Recognition')
print('='*60)
print(f'🔄 Loading model on {DEV}...')

try:
    if TOKEN:
        model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
    else:
        # Try without token (uses cached model)
        model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
    print('✅ Model loaded successfully!')
    print('🎤 Ready to transcribe Kannada speech!\n')
except Exception as e:
    print(f'❌ Failed to load model: {e}')
    if not TOKEN:
        print('💡 Tip: Create .env file with HF_TOKEN=your_token_here')
    exit(1)

def transcribe_audio(audio_path):
    """Transcribe audio using Pragna Vaani"""
    if audio_path is None:
        return "⚠️ Please upload an audio file or record speech"
    
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
        return f"❌ Error: {str(e)}"

# Create UI
with gr.Blocks(title="Pragna Vaani - Kannada Speech Recognition") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; color: #059669; margin: 0;">
            🎙️ ಪ್ರಜ್ಞಾ ವಾಣಿ
        </h1>
        <h2 style="font-size: 1.8rem; color: #374151; margin: 0;">
            Pragna Vaani
        </h2>
        <p style="font-size: 1.1rem; color: #6B7280;">
            Voice of Wisdom · Kannada Speech Recognition
        </p>
        <p style="font-size: 0.9rem; color: #9CA3AF;">
            Powered by SraVaani-1.0 · Supports 65+ Indian Languages
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            audio_input = gr.Audio(
                type="filepath",
                label="Upload Audio File or Record"
            )
            gr.Markdown("📌 Supports: WAV, MP3, M4A, FLAC")
            
            with gr.Row():
                transcribe_btn = gr.Button("🔊 Transcribe", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
            
            gr.Markdown("""
            ### 💡 Tips
            - 🎯 Best with clear audio (16kHz, mono)
            - 🗣️ Works with Kannada and 65+ Indian languages
            - ⚡ First run may take longer (model loading)
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Transcription")
            output = gr.Textbox(
                label="",
                lines=6,
                placeholder="📝 Your transcription will appear here..."
            )
            
            gr.Markdown("""
            ### 🌐 Supported Languages
            Kannada · Hindi · Tamil · Telugu · Malayalam · Bengali · Marathi · Gujarati · Odia · Punjabi · Assamese · and 50+ more
            """)
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0; margin-top: 20px; border-top: 1px solid #E5E7EB;">
        <p style="color: #6B7280; font-size: 0.9rem;">
            🌟 Pragna Vaani · Built with ❤️ for Kannada Speech Recognition
        </p>
        <p style="color: #9CA3AF; font-size: 0.8rem;">
            ARTPARK-IISc · SraVaani-1.0 · VAANI Dataset
        </p>
    </div>
    """)
    
    transcribe_btn.click(transcribe_audio, inputs=audio_input, outputs=output)
    audio_input.change(transcribe_audio, inputs=audio_input, outputs=output)
    clear_btn.click(lambda: "", outputs=output)
    clear_btn.click(lambda: None, inputs=audio_input)

if __name__ == "__main__":
    print('\n' + '='*60)
    print('🚀 Starting Pragna Vaani Web Interface...')
    print('🌐 Open: http://127.0.0.1:7860')
    print('📝 Press Ctrl+C to stop')
    print('='*60 + '\n')
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
