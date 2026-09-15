#!/usr/bin/env python3
"""
Pragna Vaani - Kannada Speech Recognition
Supports short audio (single pass) and long audio (60s chunking).
"""

import os
import subprocess
import tempfile
import time
import uuid
import numpy as np
import soundfile as sf
import torch
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoModel
from dotenv import load_dotenv

# Load token
load_dotenv()
TOKEN = os.getenv('HF_TOKEN')
DEV = 'cpu'

# This box has 8 cores but runs well above that in load average, so torch's
# default of one thread per core collapses: the OpenMP threads spin-waiting on
# each other can't all get scheduled, and every parallel step stalls on the
# slowest one. Measured on a 60s chunk: 8 threads = 34.8s, 2 threads = 7.2s.
# Override with ASR_THREADS if the machine's load profile changes.
ASR_THREADS = int(os.getenv('ASR_THREADS', '2'))
torch.set_num_threads(ASR_THREADS)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# File size limit: 500 MB (plenty for long audio)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Load Kannada ASR Model (SraVaani-1.0)
# ============================================================
print('🔄 Loading Kannada ASR model (SraVaani-1.0)...')
REPO = 'ARTPARK-IISc/SraVaani-1.0'
try:
    if TOKEN:
        asr_model = AutoModel.from_pretrained(
            REPO, trust_remote_code=True, token=TOKEN
        ).to(DEV).eval()
    else:
        asr_model = AutoModel.from_pretrained(
            REPO, trust_remote_code=True
        ).to(DEV).eval()
    print('✅ Kannada ASR model ready!')
except Exception as e:
    print(f'❌ Failed to load model: {e}')
    exit(1)

print('\n✅ All models loaded!\n')


# ============================================================
# Helper Functions
# ============================================================
def add_kannada_punctuation(text):
    """Add simple Kannada punctuation."""
    text = ' '.join(text.split())
    if text and text[-1] not in ['.', '।', '?', '!']:
        text = text + '।'
    if text.endswith('.'):
        text = text[:-1] + '।'
    return text


def cleanup_file(path):
    """Safely delete a file."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def convert_to_wav(input_path):
    """Convert any audio format to 16 kHz mono WAV."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    out_path = tmp.name
    try:
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', input_path,
            '-ac', '1',
            '-ar', '16000',
            '-acodec', 'pcm_s16le',
            out_path,
        ], capture_output=True, check=True, timeout=180)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            cleanup_file(out_path)
            return None
        return out_path
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timeout")
        cleanup_file(out_path)
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()[:200] if e.stderr else e}")
        cleanup_file(out_path)
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        cleanup_file(out_path)
        return None


def split_wav_into_chunks(data, sr, chunk_sec=60):
    """
    Split already-loaded audio into fixed-size chunks.
    Returns list of (start_sec, end_sec, waveform) with waveform as a 1-D array.

    The model's transcribe() accepts raw waveforms as well as paths, so chunks
    stay in memory instead of being written to temp WAVs and decoded back.
    """
    mono = data[:, 0] if data.ndim > 1 else data
    total_samples = len(mono)
    total_dur = total_samples / sr
    step = int(chunk_sec * sr)

    chunks = []
    for start in range(0, total_samples, step):
        piece = np.ascontiguousarray(mono[start:start + step], dtype=np.float32)
        s = start / sr
        e = min((start + step) / sr, total_dur)
        chunks.append((s, e, piece))
    return chunks


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve uploaded audio files for playback."""
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path)
    return "File not found", 404


@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    """Kannada ASR: short audio = single pass, long audio = 60s chunks."""

    if request.method == 'OPTIONS':
        r = jsonify({'status': 'ok'})
        r.headers.add('Access-Control-Allow-Origin', '*')
        r.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        r.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return r

    temp_wav = None
    uploaded_path = None

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        # Save upload
        ext = file.filename.rsplit('.', 1)[1].lower()
        fname = f"{uuid.uuid4()}.{ext}"
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(uploaded_path)
        print(f"📁 Saved: {file.filename} ({os.path.getsize(uploaded_path)} bytes)")

        # Convert to 16 kHz mono wav
        temp_wav = convert_to_wav(uploaded_path)
        if not temp_wav:
            return jsonify({'error': 'Failed to convert audio. Is ffmpeg installed?'}), 500

        # Get duration
        data, sr = sf.read(temp_wav, dtype='float32', always_2d=True)
        duration = len(data) / sr
        print(f"📏 Duration: {duration:.1f}s")

        # Route: short vs long
        if duration <= 65:
            # ---- SHORT: single pass ----
            print(f"🎤 Short file — single-pass ASR")
            text = asr_model.transcribe([temp_wav])[0]
            final_text = add_kannada_punctuation(text)
        else:
            # ---- LONG: split into 60s chunks ----
            # The exported TorchScript graph has a fixed positional-encoding
            # table (~3113 encoder frames, roughly 4 minutes), so anything
            # longer has to be chunked regardless of how much memory is free.
            chunks = split_wav_into_chunks(data, sr, chunk_sec=60)
            print(f"✂️  Split into {len(chunks)} chunks of ~60s")

            parts = []
            for i, (s, e, piece) in enumerate(chunks):
                print(f"   [{i + 1}/{len(chunks)}] {s:.1f}s - {e:.1f}s ... ",
                      end='', flush=True)
                t0 = time.time()
                try:
                    t = asr_model.transcribe([piece])[0]
                    t = add_kannada_punctuation(t)
                    print(f"ok ({time.time() - t0:.1f}s)")
                except Exception as ex:
                    t = ''
                    print(f"FAILED: {ex}")
                if t:
                    parts.append(t)
            final_text = ' '.join(parts)

        # Cleanup temp files
        cleanup_file(temp_wav)

        response = jsonify({
            'success': True,
            'result': final_text,
            'audio_url': f'/audio/{fname}',
            'duration': duration,
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        cleanup_file(temp_wav)
        cleanup_file(uploaded_path)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('🎙️ Pragna Vaani — Kannada Speech Recognition')
    print('🌐 Open: http://127.0.0.1:2000')
    print('📝 Press Ctrl+C to stop')
    print('=' * 60 + '\n')

    app.run(host='0.0.0.0', port=2000, debug=True, threaded=True)
