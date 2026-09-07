#!/bin/bash
echo "🎙️ Starting Vaani ASR Web UI..."
cd ~/Downloads/stt
source vaani_env/bin/activate
python app.py
