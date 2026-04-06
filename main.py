"""
main.py — Crate Audio Intelligence API
"""
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fetcher import analyze_song
from analyzer import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crate Audio Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    song: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.song.strip():
        raise HTTPException(status_code=400, detail="Song query cannot be empty.")
    try:
        result   = analyze_song(req.song.strip())
        features = extract_features(result["audio"], result["sr"])
        return {"metadata": result["metadata"], "features": features}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log full traceback so Railway shows it
        tb = traceback.format_exc()
        logger.error(f"Analysis error for '{req.song}':\n{tb}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n{tb}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
