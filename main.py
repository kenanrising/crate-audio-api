"""
main.py — Crate Audio Intelligence API
─────────────────────────────────────────────────────────────────────────────
POST /analyze
  Input:  { "song": "Blinding Lights The Weeknd" }
  Output: { "metadata": {...}, "features": { 20 data points } }

GET /health
  Returns: { "status": "ok" }

GET /docs
  Auto-generated interactive API docs (Swagger UI)
─────────────────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from fetcher import analyze_song
from analyzer import extract_features

app = FastAPI(
    title="Crate Audio Intelligence API",
    description="Extracts 20 audio features from any song via Apple Music preview.",
    version="1.0.0"
)

# Allow requests from the dashboard (any origin for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    song: str


class AnalyzeResponse(BaseModel):
    metadata: dict
    features: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Search for a song on Apple Music, fetch its 30-second preview,
    and extract 20 audio features.

    **Example request:**
    ```json
    { "song": "Blinding Lights The Weeknd" }
    ```

    **Example response:**
    ```json
    {
      "metadata": {
        "title": "Blinding Lights",
        "artist": "The Weeknd",
        "album": "After Hours",
        "genre": "Pop",
        "release_year": "2019",
        "artwork_url": "https://..."
      },
      "features": {
        "bpm": 171.0,
        "key": "A",
        "mode": "minor",
        "loudness_db": -8.2,
        "energy": 0.84,
        "danceability": 0.76,
        "valence": 0.32,
        "acousticness": 0.12,
        "instrumentalness": 0.03,
        "liveness": 0.08,
        "brightness_hz": 3241.0,
        "spectral_rolloff_hz": 5812.0,
        "spectral_bandwidth": 2103.0,
        "zero_crossing_rate": 0.0921,
        "harmonic_ratio": 0.71,
        "percussiveness": 0.38,
        "tempo_stability": 0.91,
        "tonal_complexity": 0.43,
        "onset_rate": 4.2,
        "duration_sec": 30.0
      }
    }
    ```
    """
    if not req.song or not req.song.strip():
        raise HTTPException(status_code=400, detail="Song query cannot be empty.")

    try:
        result   = analyze_song(req.song.strip())
        features = extract_features(result["audio"], result["sr"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return AnalyzeResponse(
        metadata=result["metadata"],
        features=features,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
