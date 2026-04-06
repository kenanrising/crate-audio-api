"""
fetcher.py
─────────────────────────────────────────────────────────────────────────────
Searches Deezer for a song and returns a 30-second mp3 preview as a numpy
array ready for feature extraction.

Deezer Search API:
 - No API key required, completely free
 - Returns direct .mp3 preview URLs (no ffmpeg/conversion needed)
 - Works on any platform, any server, any environment
─────────────────────────────────────────────────────────────────────────────
"""

import os
import tempfile
import urllib.request
import urllib.parse
import json
import warnings

import numpy as np
import librosa

warnings.filterwarnings("ignore")

SAMPLE_RATE   = 22050
CLIP_DURATION = 30.0
DEEZER_SEARCH = "https://api.deezer.com/search"


def search_song(query: str) -> dict:
    """Search Deezer for a track. Returns metadata + mp3 preview URL."""
    params = urllib.parse.urlencode({"q": query, "limit": 1})
    req = urllib.request.Request(
        f"{DEEZER_SEARCH}?{params}",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())

    results = data.get("data", [])
    if not results:
        raise ValueError(f"No results found for '{query}'. Try including the artist name.")

    t = results[0]
    preview_url = t.get("preview", "")
    if not preview_url:
        raise ValueError(f"Found '{t.get('title')}' but no preview is available for this track.")

    # Get album art (Deezer returns 56x56 by default — upgrade to 500x500)
    artwork = t.get("album", {}).get("cover_xl") or \
              t.get("album", {}).get("cover_big") or \
              t.get("album", {}).get("cover", "")

    return {
        "title":        t.get("title", "Unknown"),
        "artist":       t.get("artist", {}).get("name", "Unknown"),
        "album":        t.get("album", {}).get("title", ""),
        "genre":        "",   # Deezer search doesn't return genre; could add /track/{id} call
        "release_year": "",
        "artwork_url":  artwork,
        "preview_url":  preview_url,
        "track_id":     str(t.get("id", "")),
    }


def fetch_audio(preview_url: str) -> tuple:
    """Download mp3 preview and decode to numpy array with librosa."""
    req = urllib.request.Request(
        preview_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        audio_bytes = r.read()

    # Write mp3 to temp file — soundfile + librosa handle mp3 natively
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return y, sr


def analyze_song(query: str) -> dict:
    """Full pipeline: search → fetch preview → return metadata + audio array."""
    metadata = search_song(query)
    audio, sr = fetch_audio(metadata["preview_url"])
    return {"metadata": metadata, "audio": audio, "sr": sr}
