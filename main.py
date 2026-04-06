"""
main.py — Crate Audio Intelligence API
"""
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fetcher import analyze_song
from analyzer import extract_features

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

@app.get("/debug")
def debug():
    """Test endpoint — runs full pipeline and returns error detail"""
    try:
        result   = analyze_song("Bad Guy Billie Eilish")
        features = extract_features(result["audio"], result["sr"])
        return {"status": "ok", "bpm": features["bpm"], "key": features["key"]}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

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
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
