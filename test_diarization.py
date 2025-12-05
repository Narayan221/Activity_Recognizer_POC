from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

print(f"Token: {hf_token[:10]}...")
print("Loading pipeline...")

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    print("Pipeline loaded successfully!")
    
    audio_path = "temp_diarization_audio.wav"
    print(f"Running diarization on: {audio_path}")
    print(f"File exists: {os.path.exists(audio_path)}")
    
    if os.path.exists(audio_path):
        diarization = pipeline(audio_path)
        print("Diarization complete!")
        print(f"Type: {type(diarization)}")
        
        # Print first few tracks
        count = 0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            count += 1
            if count <= 5:
                print(f"Track {count}: {turn.start:.2f}-{turn.end:.2f} Speaker: {speaker}")
        print(f"Total tracks: {count}")
    else:
        print("Audio file not found!")
        
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())
