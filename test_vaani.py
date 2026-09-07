#!/usr/bin/env python3
"""
Vaani ASR - Quick Test Script
Tests if the model is working correctly
"""

import os
import sys
import torch
from transformers import AutoModel

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    """Print colored status messages"""
    if status == "success":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "error":
        print(f"{RED}❌ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠️ {message}{RESET}")
    else:
        print(f"{BLUE}🔄 {message}{RESET}")

def test_model():
    """Test if Vaani model loads and works"""
    
    print(f"\n{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}🧪 Vaani ASR - System Test{RESET}")
    print(f"{BLUE}{'='*50}{RESET}\n")
    
    # Test 1: Check Python version
    print_status("Checking Python version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 10:
        print_status(f"Python {python_version.major}.{python_version.minor} ✅", "success")
    else:
        print_status(f"Python {python_version.major}.{python_version.minor} - Need 3.10+", "error")
        return False
    
    # Test 2: Check if model exists in cache
    print_status("Checking if model is cached...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_exists = any("SraVaani" in f for f in os.listdir(cache_dir)) if os.path.exists(cache_dir) else False
    
    if model_exists:
        print_status("Model found in cache ✅", "success")
    else:
        print_status("Model not in cache - will download on first run", "warning")
    
    # Test 3: Check PyTorch
    print_status("Checking PyTorch...")
    try:
        import torch
        print_status(f"PyTorch {torch.__version__} ✅", "success")
    except ImportError:
        print_status("PyTorch not installed", "error")
        return False
    
    # Test 4: Check Transformers
    print_status("Checking Transformers...")
    try:
        import transformers
        print_status(f"Transformers {transformers.__version__} ✅", "success")
    except ImportError:
        print_status("Transformers not installed", "error")
        return False
    
    # Test 5: Check SoundFile
    print_status("Checking SoundFile...")
    try:
        import soundfile
        print_status("SoundFile ✅", "success")
    except ImportError:
        print_status("SoundFile not installed", "error")
        return False
    
    # Test 6: Check FFmpeg
    print_status("Checking FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("FFmpeg ✅", "success")
        else:
            print_status("FFmpeg not found", "warning")
    except FileNotFoundError:
        print_status("FFmpeg not installed (optional)", "warning")
    
    # Test 7: Load the model
    print_status("\nLoading Vaani model...")
    print_status("This may take 30-60 seconds on first run...", "info")
    
    try:
        REPO = 'ARTPARK-IISc/SraVaani-1.0'
        DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
        print_status(f"Using device: {DEV}", "info")
        
        model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()
        print_status("Model loaded successfully! ✅", "success")
        
    except Exception as e:
        print_status(f"Failed to load model: {e}", "error")
        return False
    
    # Test 8: Check available audio files
    print_status("\nLooking for test audio files...")
    
    # Check various possible audio files
    audio_files = []
    for ext in ['*.wav', '*.m4a', '*.mp3', '*.flac']:
        import glob
        audio_files.extend(glob.glob(ext))
        audio_files.extend(glob.glob(f"examples/{ext}"))
    
    if audio_files:
        print_status(f"Found {len(audio_files)} audio files: {', '.join(audio_files[:3])}{'...' if len(audio_files) > 3 else ''}", "success")
        
        # Test transcribe first file
        test_file = audio_files[0]
        print_status(f"\nTesting transcription on: {os.path.basename(test_file)}", "info")
        
        try:
            # Get transcription
            result = model.transcribe([test_file])[0]
            print_status(f"Transcription: {result}", "success")
            print_status("🎉 All tests passed! Your Vaani setup is working perfectly! 🎉", "success")
            return True
            
        except Exception as e:
            print_status(f"Transcription test failed: {e}", "error")
            return False
            
    else:
        print_status("No audio files found", "warning")
        print_status("\nYou can test with:")
        print(f"{BLUE}  python src/transcribe.py /path/to/audio.wav{RESET}")
        print_status("🎉 Setup looks good! Vaani is ready to use!", "success")
        return True

if __name__ == "__main__":
    try:
        success = test_model()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("\nTest cancelled by user", "warning")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        sys.exit(1)
