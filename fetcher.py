"""
fetcher.py — Searches iTunes for a song and returns audio as numpy array.
Uses ffmpeg subprocess to convert m4a → wav before loading with librosa.
"""

import os
import tempfile
import subprocess
import urllib.request
import urllib.parse
import json
import warnings

import numpy as np
import librosa

warnings.filterwarnings("ignore")

SAMPLE_RATE   = 22050
CLIP_DURATION = 30.0
ITUNES_SEARCH = "https://itunes.apple.com/search"


def search_song(query: str) -> dict:
    params = urllib.parse.urlencode({"term": query, "media": "music", "limit": 1, "country": "US"})
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
        raise ValueError(f"Found '{track.get('trackName')}' but no 30s preview is available.")

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


def fetch_audio(preview_url: str) -> tuple:
    """Download preview and decode to numpy array via ffmpeg."""
    # Download raw audio bytes
    req = urllib.request.Request(preview_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        audio_bytes = r.read()

    # Write to temp file
    suffix = ".m4a" if "m4a" in preview_url.lower() else ".mp3"
    tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.write(audio_bytes)
    tmp_in.close()
    tmp_wav.close()

    try:
        # Convert to wav using ffmpeg — works on any platform with ffmpeg installed
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in.name, "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-t", str(CLIP_DURATION), tmp_wav.name],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-300:]}")

        # Load the wav with librosa (soundfile handles wav perfectly)
        y, sr = librosa.load(tmp_wav.name, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    finally:
        for f in [tmp_in.name, tmp_wav.name]:
            if os.path.exists(f):
                os.unlink(f)

    return y, sr


def analyze_song(query: str) -> dict:
    metadata = search_song(query)
    audio, sr = fetch_audio(metadata["preview_url"])
    return {"metadata": metadata, "audio": audio, "sr": sr}
