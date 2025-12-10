import logging
import os
import tempfile
import subprocess
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

class WhisperService:
    """Service for audio transcription using local Whisper medium model."""
    
    def __init__(self, model_name="openai/whisper-medium"):
        """
        Initialize Whisper service with local model.
        
        Args:
            model_name: Hugging Face model name or local path to Whisper model
        """
        logging.info(f"Loading Whisper model: {model_name}")
        
        try:
            # Determine device
            device_id = 0 if torch.cuda.is_available() else -1
            self.device = "cuda" if device_id == 0 else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Explicitly load model and processor to avoid "meta tensor" errors with pipeline
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(model_name)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device_id
            )
            
            logging.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise
    
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
        Transcribe complete audio from video file with proper timestamps.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        audio_path = None
        
        try:
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Transcribe using pipeline (handles long audio automatically)
            logging.info("Starting transcription...")
            result = self.pipe(
                audio_path,
                return_timestamps=True,
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                    # Anti-repetition settings
                    "no_repeat_ngram_size": 2,  # Prevent 2-gram repetitions
                    "temperature": 0.0,         # Use greedy decoding by default
                    "do_sample": False          # Deterministic output
                }
            )
            
            # Extract full text
            full_text = result.get("text", "")
            
            # Extract segments with timestamps
            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    segments.append({
                        "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                        "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                        "text": chunk["text"].strip()
                    })
            else:
                # Fallback if no chunks
                segments = [{
                    "start": 0.0,
                    "end": 0.0,
                    "text": full_text
                }]
            
            response = {
                "language": result.get("language", "en"),
                "text": full_text,
                "segments": segments,
                "total_segments": len(segments)
            }
            
            logging.info(f"Transcription completed: {len(segments)} segments, {len(full_text)} characters")
            return response
            
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return {
                "language": "unknown",
                "text": "",
                "segments": [],
                "total_segments": 0,
                "error": str(e)
            }
        finally:
            # Clean up audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logging.info("Cleaned up temporary audio file")
                except Exception:
                    pass
