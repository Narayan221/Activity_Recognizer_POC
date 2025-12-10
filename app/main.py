import logging
from dotenv import load_dotenv

# Load env vars before importing anything else
load_dotenv()
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from app.api.routes import router as video_router

app = FastAPI(title="Human Activity Recognizer API", version="1.0")

app.include_router(video_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Human Activity Recognizer API. Docs at /docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
