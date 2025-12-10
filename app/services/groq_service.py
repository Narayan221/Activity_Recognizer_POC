import logging
import os
import tempfile
import subprocess
from groq import Groq

class GroqService:
    """Service for audio transcription using Groq API."""
    
    def __init__(self, api_key=None):
        """
        Initialize Groq service.
        
        Args:
            api_key: Groq API key (optional, reads from GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logging.warning("GROQ_API_KEY not found. Groq transcription will fail.")
        
        try:
            self.client = Groq(api_key=self.api_key)
            logging.info("Groq client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            self.client = None

    def extract_audio(self, video_path):
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file (WAV format)
        """
        # Create temporary audio file
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_path = audio_file.name
        audio_file.close()
        
        try:
            # Use ffmpeg to extract audio
            # -vn: no video, -acodec pcm_s16le: WAV format, -ar 16000: 16kHz sample rate
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y',  # Overwrite output file
                audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise Exception("Failed to extract audio from video")
        except Exception as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise

    def transcribe(self, video_path):
        """
        Transcribe audio from video file using Groq Whisper.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing transcription results
        """
        audio_path = None
        
        try:
            if not self.client:
                return {"error": "Groq client not initialized", "text": "", "segments": []}
            
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            logging.info("Starting Groq transcription...")
            
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            
            # Parse response
            # Groq's verbose_json returns an object with text and segments
            full_text = transcription.text
            segments = []
            
            if hasattr(transcription, 'segments'):
                for seg in transcription.segments:
                    # Handle segment as object or dict
                    start = seg.start if hasattr(seg, 'start') else seg.get('start', 0.0)
                    end = seg.end if hasattr(seg, 'end') else seg.get('end', 0.0)
                    text = seg.text if hasattr(seg, 'text') else seg.get('text', "")
                    
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text.strip()
                    })
            else:
                # Fallback
                segments = [{"start": 0.0, "end": 0.0, "text": full_text}]
            
            response = {
                "language": transcription.language if hasattr(transcription, 'language') else "en",
                "text": full_text,
                "segments": segments,
                "service": "groq-whisper-large-v3"
            }
            
            logging.info(f"Groq transcription completed: {len(segments)} segments")
            return response
            
        except Exception as e:
            logging.error(f"Groq transcription failed: {e}")
            return {
                "language": "unknown",
                "text": "",
                "segments": [],
                "error": str(e)
            }
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
