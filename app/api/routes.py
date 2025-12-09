from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
import shutil
from app.services.video_processor import VideoProcessor

router = APIRouter()
processor = VideoProcessor()

@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file format")
        
    # Use a temporary file to store the upload
    # delete=False is required for Windows to open the file via cv2 while it still exists
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    temp_path = temp_file.name
    
    try:
        # Write upload content to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the video
        # We pass the original filename so the report looks correct
        result = processor.process(temp_path, original_filename=file.filename)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure the temp file is removed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass # Best effort cleanup
