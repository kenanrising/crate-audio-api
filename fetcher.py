"""
fetcher.py
─────────────────────────────────────────────────────────────────────────────
Searches for a song by name using the iTunes Search API and returns
a 30-second audio preview as a numpy array ready for feature extraction.

iTunes Search API:
 - No API key required
 - Returns 30s .m4a previews for 100M+ songs
 - Covers all major artists across every genre
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

SAMPLE_RATE   = 22050   # librosa default — good balance of quality vs speed
CLIP_DURATION = 30.0    # full iTunes preview

ITUNES_SEARCH = "https://itunes.apple.com/search"


def search_song(query: str) -> dict:
    """
    Search iTunes for a song. Returns metadata dict with preview_url.
    Raises ValueError if nothing found or no preview available.
    """
    params = urllib.parse.urlencode({
        "term":    query,
        "media":   "music",
        "limit":   1,
        "country": "US",
    })
    req = urllib.request.Request(
        f"{ITUNES_SEARCH}?{params}",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())

    if data.get("resultCount", 0) == 0:
        raise ValueError(f"No results found for '{query}'. Try including the artist name.")

    track = data["results"][0]
    preview_url = track.get("previewUrl")

    if not preview_url:
        raise ValueError(
            f"Found '{track.get('trackName')}' but Apple Music has no 30s preview for this track. "
            "Try a different song."
        )

    return {
        "title":        track.get("trackName", "Unknown"),
        "artist":       track.get("artistName", "Unknown"),
        "album":        track.get("collectionName", ""),
        "genre":        track.get("primaryGenreName", ""),
        "release_year": str(track.get("releaseDate", ""))[:4],
        "artwork_url":  track.get("artworkUrl100", "").replace("100x100", "400x400"),
        "preview_url":  preview_url,
        "track_id":     str(track.get("trackId", "")),
    }


def fetch_audio(preview_url: str) -> tuple[np.ndarray, int]:
    """
    Download a 30s preview URL and return (audio_array, sample_rate).
    """
    req = urllib.request.Request(
        preview_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        audio_bytes = r.read()

    suffix = ".m4a" if "m4a" in preview_url else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return y, sr


def analyze_song(query: str) -> dict:
    """
    Full pipeline: search → fetch preview → return metadata + raw audio.
    Returns: {"metadata": {...}, "audio": np.ndarray, "sr": int}
    """
    metadata = search_song(query)
    audio, sr = fetch_audio(metadata["preview_url"])
    return {"metadata": metadata, "audio": audio, "sr": sr}
