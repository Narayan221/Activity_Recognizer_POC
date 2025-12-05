import whisper
import sys
import os
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

def transcribe_video_audio(video_path, model_size='base', output_file=None):
    """
    Extract and transcribe audio from video file.
    
    Args:
        video_path: Path to video file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_file: Optional output text file path
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Set default output filename
    if output_file is None:
        base_name = os.path.splitext(video_path)[0]
        output_file = f"{base_name}_transcript.txt"
    
    print("=" * 80)
    print("AUDIO TRANSCRIPTION")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Model: {model_size}")
    print(f"Output: {output_file}")
    print("=" * 80)
    
    # Load Whisper model
    print(f"\nLoading Whisper model ({model_size})...")
    print("(First time will download the model, please wait...)")
    model = whisper.load_model(model_size)
    
    # Transcribe
    print("\nTranscribing audio... This may take a few minutes.")
    print("(Whisper will extract audio automatically)")
    
    try:
        result = model.transcribe(video_path, verbose=True)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("AUDIO TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Source: {video_path}\n")
            f.write(f"Model: {model_size}\n")
            f.write(f"Language: {result.get('language', 'unknown')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Full text
            f.write("FULL TRANSCRIPT:\n")
            f.write("-" * 80 + "\n")
            f.write(result['text'] + "\n\n")
            
            # Timestamped segments
            f.write("=" * 80 + "\n")
            f.write("TIMESTAMPED SEGMENTS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"[{i}] {start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRANSCRIPTION COMPLETE!")
        print("=" * 80)
        print(f"Total segments: {len(result['segments'])}")
        print(f"Language detected: {result.get('language', 'unknown')}")
        print(f"Output saved to: {output_file}")
        print("=" * 80)
        
        # Print first few lines
        print("\nPreview (first 500 characters):")
        print("-" * 80)
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
        print("-" * 80)
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure FFmpeg is installed and in PATH")
        print("2. Restart your terminal after installing FFmpeg")
        print("3. Try with a different video file")
        print("4. Check if the video has audio")
        return
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("AUDIO TRANSCRIPTION TOOL")
        print("=" * 80)
        print("\nUsage:")
        print("  python audio_transcriber.py <video_file> [model_size] [output_file]")
        print("\nExamples:")
        print("  python audio_transcriber.py video.mp4")
        print("  python audio_transcriber.py video.mp4 medium")
        print("  python audio_transcriber.py video.mp4 medium transcript.txt")
        print("\nWhisper Models:")
        print("  tiny   - Fastest, least accurate (~1GB RAM)")
        print("  base   - Fast, good for testing (~1GB RAM) [DEFAULT]")
        print("  small  - Balanced (~2GB RAM)")
        print("  medium - Recommended for quality (~5GB RAM)")
        print("  large  - Best quality, slowest (~10GB RAM)")
        print("\nNote: Requires FFmpeg to be installed!")
        print("=" * 80)
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else 'base'
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    transcribe_video_audio(video_path, model_size, output_file)
