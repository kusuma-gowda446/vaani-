#!/usr/bin/env python3
"""
Pragna Vaani - Flask Web UI
Kannada Speech Recognition & English-to-Kannada Translation
"""

import os
import subprocess
import tempfile
import uuid
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import whisper
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

# Load token
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')
DEV = 'cpu'

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Allowed extensions
ALLOWED_EXTENSIONS = {
    'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus', 
    'mp4', 'mpeg', 'mpga', 'webm', 'm4p', 'm4b', 'm4r',
    '3gp', '3gpp', '3g2', 'amr', 'aiff', 'aif', 'aifc',
    'au', 'snd', 'raw', 'pcm', 'caf'
}

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# ============================================================
# Load Models
# ============================================================

print('🔄 Loading models...')

print('  → Kannada ASR (SraVaani-1.0)...')
REPO = 'ARTPARK-IISc/SraVaani-1.0'
if TOKEN:
    asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
else:
    asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
print('  ✅ Kannada ASR ready!')

print('  → English ASR (Whisper)...')
whisper_model = whisper.load_model("base")
print('  ✅ Whisper ready!')

print('  → Translation (NLLB)...')
trans_model_name = "facebook/nllb-200-distilled-600M"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(DEV).eval()
print('  ✅ Translation ready!')

print('\n✅ All models loaded!\n')

# ============================================================
# Helper Functions
# ============================================================

def add_kannada_punctuation(text):
    text = ' '.join(text.split())
    if text and not text[-1] in ['.', '।', '?', '!']:
        text = text + '।'
    if text.endswith('.'):
        text = text[:-1] + '।'
    return text

def convert_to_wav(input_path):
    try:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ac', '1', '-ar', '16000',
            '-acodec', 'pcm_s16le',
            temp_wav, '-y', '-loglevel', 'error'
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return temp_wav
    except:
        return None

def cleanup_file(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass

# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve uploaded audio files for playback"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav')
    return "File not found", 404

@app.route('/transcribe', methods=['POST'])
def transcribe():
    temp_file = None
    uploaded_file_path = None
    
    try:
        mode = request.form.get('mode', 'kannada')
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get extension
        original_filename = file.filename
        ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        
        if not allowed_file(original_filename):
            return jsonify({'error': f'File type "{ext}" not supported'}), 400
        
        # Save file
        filename = str(uuid.uuid4()) + '.' + (ext if ext else 'wav')
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        
        # Convert to WAV if needed
        if ext != 'wav':
            print(f'🔄 Converting {ext} to WAV...')
            temp_file = convert_to_wav(uploaded_file_path)
            if not temp_file:
                cleanup_file(uploaded_file_path)
                return jsonify({'error': 'Failed to convert audio'}), 500
            audio_to_process = temp_file
        else:
            audio_to_process = uploaded_file_path
        
        # Process
        if mode == 'kannada':
            print(f'🎤 Kannada ASR: {os.path.basename(audio_to_process)}')
            result = asr_model.transcribe([audio_to_process])[0]
            result = add_kannada_punctuation(result)
            
            cleanup_file(temp_file)
            
            return jsonify({
                'success': True,
                'result': result,
                'mode': 'Kannada ASR',
                'audio_url': f'/audio/{filename}'
            })
        
        elif mode == 'english':
            print(f'🌐 English → Kannada: {os.path.basename(audio_to_process)}')
            
            # Transcribe English
            result = whisper_model.transcribe(audio_to_process, language="en")
            english_text = result["text"]
            
            # Translate to Kannada
            inputs = trans_tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = translation_model.generate(
                    **inputs,
                    forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids("kan_Knda"),
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            kannada_text = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            cleanup_file(temp_file)
            
            return jsonify({
                'success': True,
                'english': english_text,
                'result': kannada_text,
                'mode': 'English → Kannada',
                'audio_url': f'/audio/{filename}'
            })
        
        else:
            cleanup_file(temp_file)
            cleanup_file(uploaded_file_path)
            return jsonify({'error': f'Invalid mode: {mode}'}), 400
    
    except Exception as e:
        cleanup_file(temp_file)
        cleanup_file(uploaded_file_path)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('\n' + '='*60)
    print('🚀 Pragna Vaani - Flask UI')
    print('🌐 Open: http://127.0.0.1:5000')
    print('📝 Press Ctrl+C to stop')
    print('='*60 + '\n')
    app.run(host='127.0.0.1', port=5000, debug=True)
