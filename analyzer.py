"""
analyzer.py
─────────────────────────────────────────────────────────────────────────────
Extracts 20 audio features from a raw audio array using librosa.
All features are normalized to human-readable ranges where possible.
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MINOR_TEMPLATE = np.array([1,0,0,1,0,1,0,1,0,0,1,0], dtype=float)
MAJOR_TEMPLATE = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Given a mono audio array and sample rate, return all 20 features as a dict.
    All float values are rounded to 4 decimal places for clean JSON output.
    """

    # ── Preprocessing ─────────────────────────────────────────────────────────
    y = y.astype(np.float32)
    y_harm, y_perc = librosa.effects.hpss(y)

    # Precompute reusable transforms
    S       = np.abs(librosa.stft(y))
    freqs   = librosa.fft_frequencies(sr=sr)
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr)
    rms     = librosa.feature.rms(y=y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # ── 1. BPM ────────────────────────────────────────────────────────────────
    bpm = round(float(np.squeeze(tempo)), 2)

    # ── 2. Key ────────────────────────────────────────────────────────────────
    chroma_mean = np.mean(chroma, axis=1)
    key_idx     = int(np.argmax(chroma_mean))
    key         = KEYS[key_idx]

    # ── 3. Mode (major / minor) ───────────────────────────────────────────────
    minor_score = np.dot(chroma_mean, np.roll(MINOR_TEMPLATE, key_idx))
    major_score = np.dot(chroma_mean, np.roll(MAJOR_TEMPLATE, key_idx))
    mode        = "major" if major_score >= minor_score else "minor"

    # ── 4. Loudness (dBFS RMS) ────────────────────────────────────────────────
    rms_mean  = float(np.mean(rms))
    loudness_db = round(float(20 * np.log10(rms_mean + 1e-9)), 2)

    # ── 5. Energy (0–1 normalized) ────────────────────────────────────────────
    raw_energy = float(np.sum(y**2) / len(y))
    energy     = round(min(float(raw_energy / 0.5), 1.0), 4)

    # ── 6. Danceability (beat strength, 0–1) ─────────────────────────────────
    onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
    beat_str    = float(np.mean(onset_env))
    danceability = round(min(beat_str / 5.0, 1.0), 4)

    # ── 7. Valence (0=sad/minor, 1=happy/major) ───────────────────────────────
    valence = round(float(major_score / (major_score + minor_score + 1e-6)), 4)

    # ── 8. Acousticness (0=electronic, 1=acoustic) ────────────────────────────
    low_energy  = float(np.mean(S[freqs < 2000, :]))
    high_energy = float(np.mean(S[freqs >= 2000, :]))
    acousticness = round(float(low_energy / (low_energy + high_energy + 1e-6)), 4)

    # ── 9. Instrumentalness (0=vocal, 1=instrumental) ─────────────────────────
    # High RMS variance = dynamic vocals; low variance = steady instrumental
    rms_var          = float(np.var(rms))
    instrumentalness = round(max(0.0, 1.0 - min(rms_var * 500, 1.0)), 4)

    # ── 10. Liveness (studio=0, live=1) ──────────────────────────────────────
    sf       = librosa.feature.spectral_flatness(y=y)
    liveness = round(min(float(np.mean(sf)) * 20, 1.0), 4)

    # ── 11. Spectral Centroid (brightness, Hz) ────────────────────────────────
    sc         = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = round(float(np.mean(sc)), 2)

    # ── 12. Spectral Rolloff (Hz) ─────────────────────────────────────────────
    rolloff = round(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 2)

    # ── 13. Spectral Bandwidth ────────────────────────────────────────────────
    bandwidth = round(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))), 2)

    # ── 14. Zero Crossing Rate (percussiveness / noise) ───────────────────────
    zcr = round(float(np.mean(librosa.feature.zero_crossing_rate(y))), 4)

    # ── 15. Harmonic Ratio (0–1, melody strength) ────────────────────────────
    harm_ratio = round(float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y)) + 1e-6)), 4)

    # ── 16. Percussiveness (0–1) ──────────────────────────────────────────────
    perc_ratio = round(float(np.mean(np.abs(y_perc)) / (np.mean(np.abs(y)) + 1e-6)), 4)

    # ── 17. Tempo Stability (0=unstable, 1=perfectly steady) ─────────────────
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        ibi_std    = float(np.std(np.diff(beat_times)))
        stability  = round(max(0.0, 1.0 - min(ibi_std * 5, 1.0)), 4)
    else:
        stability = 0.0

    # ── 18. Tonal Complexity (chroma variance, 0–1) ────────────────────────────
    chroma_var      = float(np.mean(np.var(chroma, axis=1)))
    tonal_complexity = round(min(chroma_var * 10, 1.0), 4)

    # ── 19. Onset Rate (events per second) ────────────────────────────────────
    onsets     = librosa.onset.onset_detect(y=y, sr=sr)
    duration   = librosa.get_duration(y=y, sr=sr)
    onset_rate = round(len(onsets) / max(duration, 1), 2)

    # ── 20. Duration (seconds) ────────────────────────────────────────────────
    duration_s = round(duration, 2)

    return {
        "bpm":               bpm,
        "key":               key,
        "mode":              mode,
        "loudness_db":       loudness_db,
        "energy":            energy,
        "danceability":      danceability,
        "valence":           valence,
        "acousticness":      acousticness,
        "instrumentalness":  instrumentalness,
        "liveness":          liveness,
        "brightness_hz":     brightness,
        "spectral_rolloff_hz": rolloff,
        "spectral_bandwidth": bandwidth,
        "zero_crossing_rate": zcr,
        "harmonic_ratio":    harm_ratio,
        "percussiveness":    perc_ratio,
        "tempo_stability":   stability,
        "tonal_complexity":  tonal_complexity,
        "onset_rate":        onset_rate,
        "duration_sec":      duration_s,
    }
