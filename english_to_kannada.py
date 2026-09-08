#!/usr/bin/env python3
"""
English to Kannada Speech Translation
Speak in English → Get Kannada Text
"""

import os
import whisper
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import subprocess
import tempfile

class EnglishToKannadaTranslator:
    def __init__(self):
        """Initialize the ASR and Translation models"""
        print("🔄 Loading English ASR model (Whisper)...")
        self.asr_model = whisper.load_model("base")  # Use "base" or "small" for speed
        
        print("🔄 Loading English-to-Kannada Translation model...")
        # Use NLLB-200 model fine-tuned for English-Kannada
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("✅ Models loaded successfully!")

    def transcribe_english(self, audio_path):
        """
        Step 1: Transcribe English speech to English text using Whisper
        """
        try:
            # Load audio with whisper
            result = self.asr_model.transcribe(audio_path)
            english_text = result["text"]
            print(f"📝 Transcribed English: {english_text}")
            return english_text
        except Exception as e:
            return f"Error in transcription: {e}"

    def translate_to_kannada(self, english_text):
        """
        Step 2: Translate English text to Kannada text using NLLB
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id["kan_Knda"],
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode output
            kannada_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"📝 Translated to Kannada: {kannada_text}")
            return kannada_text
        except Exception as e:
            return f"Error in translation: {e}"

    def translate(self, audio_path):
        """
        Complete pipeline: English Speech → Kannada Text
        """
        print("\n" + "="*60)
        print("🎙️ English → Kannada Speech Translation")
        print("="*60)
        
        # Convert audio if not WAV
        if not audio_path.lower().endswith('.wav'):
            print("🔄 Converting audio to WAV format...")
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000',
                temp_wav, '-y', '-loglevel', 'error'
            ], check=True)
            audio_path = temp_wav
        
        # Step 1: Transcribe English speech to English text
        print("\n🔊 Step 1: Transcribing English speech...")
        english_text = self.transcribe_english(audio_path)
        
        if "Error" in english_text:
            return english_text
        
        # Step 2: Translate English to Kannada
        print("\n🔄 Step 2: Translating to Kannada...")
        kannada_text = self.translate_to_kannada(english_text)
        
        # Clean up
        if 'temp_wav' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        
        print("\n" + "="*60)
        print("✅ Final Result:")
        print(f"   English: {english_text}")
        print(f"   Kannada: {kannada_text}")
        print("="*60)
        
        return kannada_text

def main():
    """Main function"""
    print("🎙️ English to Kannada Speech Translation")
    print("="*60)
    
    # Initialize translator
    translator = EnglishToKannadaTranslator()
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("Enter audio file path: ")
    
    # Translate
    result = translator.translate(audio_path)
    print(f"\n📝 Final Kannada Text: {result}")

if __name__ == "__main__":
    import sys
    main()
