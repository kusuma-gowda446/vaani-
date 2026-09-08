#!/usr/bin/env python3
"""
Pragna Vaani - Kannada Speech Recognition (Kannada Only)
Only Kannada Speech → Kannada Text
"""

import os
import subprocess
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoModel
from dotenv import load_dotenv

# Load token
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')
DEV = 'cpu'

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# File size limit
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Create folders
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {
    'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus', 
    'mp4', 'mpeg', 'mpga', 'webm', 'm4p', 'm4b', 'm4r',
    '3gp', '3gpp', '3g2', 'amr', 'aiff', 'aif', 'aifc'
}

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# ============================================================
# Load Kannada ASR Model Only
# ============================================================

print('🔄 Loading Kannada ASR model (SraVaani-1.0)...')
REPO = 'ARTPARK-IISc/SraVaani-1.0'
try:
    if TOKEN:
        asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
    else:
        asr_model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
    print('✅ Kannada ASR model ready!')
except Exception as e:
    print(f'❌ Failed to load model: {e}')
    exit(1)

print('\n✅ All models loaded!\n')

# ============================================================
# Helper Functions
# ============================================================

def add_kannada_punctuation(text):
    """Add simple punctuation to Kannada text"""
    text = ' '.join(text.split())
    if text and not text[-1] in ['.', '।', '?', '!']:
        text = text + '।'
    if text.endswith('.'):
        text = text[:-1] + '।'
    return text

def cleanup_file(file_path):
    """Clean up temporary file"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted: {os.path.basename(file_path)}")
        except:
            pass

def convert_to_wav(input_path):
    """Convert any audio format to WAV (16kHz, mono)"""
    try:
        print(f"🔄 Converting: {os.path.basename(input_path)}")
        
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        temp_wav_path = temp_wav.name
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ac', '1',
            '-ar', '16000',
            '-acodec', 'pcm_s16le',
            '-y',
            temp_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr[:200]}")
            cleanup_file(temp_wav_path)
            return None
        
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            print("❌ Output file is empty or missing")
            cleanup_file(temp_wav_path)
            return None
        
        # Verify with soundfile
        try:
            import soundfile as sf
            data, sr = sf.read(temp_wav_path, dtype='float32', always_2d=True)
            print(f"✅ Converted: {len(data)/sr:.1f}s, {sr}Hz, {data.shape[1]} channels")
        except Exception as e:
            print(f"❌ Soundfile validation failed: {e}")
            cleanup_file(temp_wav_path)
            return None
        
        return temp_wav_path
        
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg conversion timed out (60s)")
        cleanup_file(temp_wav_path)
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        cleanup_file(temp_wav_path)
        return None

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

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    """Kannada ASR: Kannada Speech → Kannada Text"""
    
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response
    
    temp_file = None
    uploaded_file_path = None
    
    try:
        # Only Kannada mode
        mode = request.form.get('mode', 'kannada')
        
        if mode != 'kannada':
            return jsonify({'error': 'Only Kannada ASR is supported. Please select Kannada mode.'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        original_filename = file.filename
        ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'webm'
        
        if not allowed_file(original_filename):
            return jsonify({'error': f'File type "{ext}" not supported'}), 400
        
        filename = str(uuid.uuid4()) + '.' + ext
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        
        print(f"📁 Saved: {original_filename} ({os.path.getsize(uploaded_file_path)} bytes)")
        
        # Convert to WAV
        print(f'🔄 Converting to WAV...')
        temp_file = convert_to_wav(uploaded_file_path)
        
        if not temp_file:
            cleanup_file(uploaded_file_path)
            return jsonify({'error': 'Failed to convert audio. Make sure ffmpeg is installed.'}), 500
        
        # Transcribe Kannada
        print(f'🎤 Kannada ASR: {os.path.basename(temp_file)}')
        result = asr_model.transcribe([temp_file])[0]
        result = add_kannada_punctuation(result)
        
        # Clean up
        cleanup_file(temp_file)
        cleanup_file(uploaded_file_path)
        
        response = jsonify({
            'success': True,
            'result': result,
            'mode': 'Kannada ASR',
            'audio_url': f'/audio/{filename}'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    except Exception as e:
        cleanup_file(temp_file)
        cleanup_file(uploaded_file_path)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('\n' + '='*60)
    print('🎙️ Pragna Vaani - Kannada Speech Recognition')
    print('🌐 Open: http://127.0.0.1:5000')
    print('📝 Press Ctrl+C to stop')
    print('='*60 + '\n')
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
