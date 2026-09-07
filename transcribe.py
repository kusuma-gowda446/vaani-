import sys
import torch
from transformers import AutoModel

# YOUR TOKEN - IMPORTANT: Replace this if you create a new token
# Token removed - use huggingface-cli login instead
REPO = 'ARTPARK-IISc/SraVaani-1.0'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'🔄 Loading model on {DEV}...')
model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=TOKEN).to(DEV).eval()
print('✅ Model loaded!')

if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = input('🎤 Enter audio file path: ')

print(f'📝 Transcribing: {audio_file}')
result = model.transcribe([audio_file])[0]
print(f'\n✅ Transcription: {result}')
