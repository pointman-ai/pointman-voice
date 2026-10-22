"""Kokoro's English voices: 20 American and 8 British, best first by Kokoro's own overall grades.

Grades from hexgrad/Kokoro-82M's VOICES.md. Voices of one grade keep that file's order.
"""

from __future__ import annotations

# (id, overall grade), best first
GRADED = [
    ("af_heart", "A"), ("af_bella", "A-"),
    ("af_nicole", "B-"), ("bf_emma", "B-"),
    ("af_aoede", "C+"), ("af_kore", "C+"), ("af_sarah", "C+"),
    ("am_fenrir", "C+"), ("am_michael", "C+"), ("am_puck", "C+"),
    ("af_alloy", "C"), ("af_nova", "C"), ("bf_isabella", "C"), ("bm_fable", "C"), ("bm_george", "C"),
    ("af_sky", "C-"),
    ("bm_lewis", "D+"),
    ("af_jessica", "D"), ("af_river", "D"), ("am_echo", "D"), ("am_eric", "D"), ("am_liam", "D"),
    ("am_onyx", "D"), ("bf_alice", "D"), ("bf_lily", "D"), ("bm_daniel", "D"),
    ("am_santa", "D-"),
    ("am_adam", "F+"),
]

LANGUAGES = {"a": "en-US", "b": "en-GB"}
GENDERS = {"f": "female", "m": "male"}


def describe(voice_id: str, grade: str | None = None) -> dict:
    """One voice as the picker shows it: af_heart → Heart, en-US, female."""
    return {"id": voice_id, "name": voice_id.split("_", 1)[1].capitalize(),
            "language": LANGUAGES[voice_id[0]], "gender": GENDERS[voice_id[1]],
            **({"grade": grade} if grade else {})}


def catalogue() -> list[dict]:
    return [describe(v, g) for v, g in GRADED]


def known(voice_id: str) -> bool:
    return any(v == voice_id for v, _ in GRADED)
