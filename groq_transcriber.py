from groq import Groq
import sys
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

def transcribe_with_groq(file_path, api_key=None, output_file=None):
    """
    Transcribe audio/video using Groq's Whisper API.
    Groq can handle both audio and video files directly.
    
    Args:
        file_path: Path to audio or video file
        api_key: Groq API key (optional if set in .env)
        output_file: Optional output text file path
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Get API key from .env if not provided
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("Error: GROQ_API_KEY not found in .env file")
            print("Please create a .env file with: GROQ_API_KEY=your_key_here")
            return
    
    # Set default output filename
    if output_file is None:
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}_transcript.txt"
    
    print("=" * 80)
    print("GROQ WHISPER TRANSCRIPTION")
    print("=" * 80)
    print(f"Input: {file_path}")
    print(f"Output: {output_file}")
    print("=" * 80)
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    print("\nTranscribing with Groq Whisper (ultra-fast)...")
    print("Uploading file to Groq...")
    
    try:
        # Open file and send to Groq
        with open(file_path, "rb") as file:
            # Create transcription - Groq handles both audio and video
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                temperature=0.0
            )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("AUDIO TRANSCRIPT (Groq Whisper Large V3)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Source: {file_path}\n")
            f.write(f"Model: whisper-large-v3\n")
            f.write(f"Language: {transcription.language}\n")
            f.write("=" * 80 + "\n\n")
            
            # Full text
            f.write("FULL TRANSCRIPT:\n")
            f.write("-" * 80 + "\n")
            f.write(transcription.text + "\n\n")
            
            # Timestamped segments if available
            if hasattr(transcription, 'segments') and transcription.segments:
                f.write("=" * 80 + "\n")
                f.write("TIMESTAMPED SEGMENTS\n")
                f.write("=" * 80 + "\n\n")
                
                for i, segment in enumerate(transcription.segments, 1):
                    start_time = format_timestamp(segment['start'])
                    end_time = format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    
                    f.write(f"[{i}] {start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRANSCRIPTION COMPLETE!")
        print("=" * 80)
        if hasattr(transcription, 'segments'):
            print(f"Total segments: {len(transcription.segments)}")
        print(f"Language detected: {transcription.language}")
        print(f"Output saved to: {output_file}")
        print("=" * 80)
        
        # Print preview
        print("\nPreview (first 500 characters):")
        print("-" * 80)
        preview_text = transcription.text[:500] + "..." if len(transcription.text) > 500 else transcription.text
        print(preview_text)
        print("-" * 80)
        
        return output_file
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Groq API key in .env file")
        print("2. Make sure the file is a valid audio/video file")
        print("3. Check your internet connection")
        print("4. File size limit: 25MB for Groq API")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("GROQ WHISPER TRANSCRIPTION TOOL")
        print("=" * 80)
        print("\nUsage:")
        print("  python groq_transcriber.py <video_or_audio_file> [output_file]")
        print("\nExamples:")
        print("  python groq_transcriber.py video.mp4")
        print("  python groq_transcriber.py video.mp4 transcript.txt")
        print("  python groq_transcriber.py audio.mp3")
        print("\nAPI Key Setup:")
        print("  Create a .env file with:")
        print("  GROQ_API_KEY=your_key_here")
        print("\nSupported formats:")
        print("  Video: mp4, avi, mov, mkv, flv, wmv")
        print("  Audio: mp3, wav, m4a, flac, ogg, webm")
        print("\nGet your Groq API key from: https://console.groq.com/keys")
        print("\nAdvantages of Groq:")
        print("  ✓ Ultra-fast transcription (10-100x faster than local)")
        print("  ✓ Handles video files directly (no audio extraction needed)")
        print("  ✓ Uses whisper-large-v3 (most accurate)")
        print("  ✓ Cloud-based (no GPU needed)")
        print("  ✓ File size limit: 25MB")
        print("=" * 80)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    transcribe_with_groq(input_file, output_file=output_file)
