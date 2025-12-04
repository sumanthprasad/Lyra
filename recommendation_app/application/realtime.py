import uuid
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .model import generate_playlist_feature

# Features available in complete_feature.csv; we keep the list explicit so feedback
# adjustments remain predictable if the dataset changes column order.
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "duration_ms_x",
]

# Simple emotion -> target feature hints (valence/energy) to nudge the profile.
EMOTION_HINTS = {
    "happy": {"valence": 0.85, "energy": 0.7, "danceability": 0.7},
    "sad": {"valence": 0.2, "energy": 0.25, "acousticness": 0.6},
    "calm": {"valence": 0.55, "energy": 0.2, "acousticness": 0.7},
    "angry": {"valence": 0.2, "energy": 0.9, "loudness": 0.8},
    "excited": {"valence": 0.75, "energy": 0.9, "tempo": 0.7},
    "romantic": {"valence": 0.65, "energy": 0.35, "acousticness": 0.55},
    "nostalgic": {"valence": 0.45, "energy": 0.35, "acousticness": 0.6},
}

# Skip reasons used to dampen or boost parts of the user profile on the fly.
REASON_BIASES = {
    "wrong_energy": {"energy": -0.2, "danceability": -0.1},
    "wrong_mood": {"valence": -0.2},
    "wrong_genre": {"speechiness": -0.1, "instrumentalness": 0.1},
    "dont_like": {"loudness": -0.05},
}


SessionStore = Dict[str, Dict[str, object]]
SESSION_STATE: SessionStore = {}


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric feature columns have no NaNs."""
    clean = df.copy()
    for col in FEATURE_COLUMNS:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)
        else:
            # Make sure the column exists so downstream slices stay aligned
            clean[col] = 0.0
    return clean


def _mean_feature_vector(feature_df: pd.DataFrame, ids: List[str]) -> Optional[pd.Series]:
    if not ids:
        return None
    subset = feature_df[feature_df["id"].isin(ids)]
    if subset.empty:
        return None
    usable_cols = [c for c in FEATURE_COLUMNS if c in subset.columns]
    mean_vec = subset[usable_cols].mean()
    return mean_vec.fillna(0.0)


def _apply_emotion_hint(profile: pd.Series, emotion: Optional[str]) -> pd.Series:
    if not emotion:
        return profile
    hint = EMOTION_HINTS.get(emotion.lower())
    if not hint:
        return profile
    for feature, target in hint.items():
        if feature in profile:
            profile[feature] = (profile[feature] * 0.6) + (target * 0.4)
    return profile


def _apply_skip_reason(profile: pd.Series, reason: Optional[str]) -> pd.Series:
    if not reason:
        return profile
    deltas = REASON_BIASES.get(reason)
    if not deltas:
        return profile
    for feature, delta in deltas.items():
        if feature in profile:
            profile[feature] = max(0.0, profile[feature] + delta)
    return profile


def _build_profile(
    base_vector: pd.Series, feedback: Dict[str, List], feature_df: pd.DataFrame
) -> pd.Series:
    profile = base_vector.copy().fillna(0.0)

    positive_ids: List[str] = []
    negative_ids: List[str] = []

    positive_ids.extend(feedback.get("liked", []))
    positive_ids.extend(feedback.get("replays", []))
    positive_ids.extend(feedback.get("pivot_ids", []))

    negative_ids.extend(feedback.get("disliked", []))

    for listen in feedback.get("listen_events", []):
        track_id = listen.get("track_id")
        seconds = listen.get("seconds", 0)
        if not track_id:
            continue
        if seconds >= 90:
            positive_ids.append(track_id)
        elif seconds <= 30:
            negative_ids.append(track_id)

    if positive_ids:
        pos_vec = _mean_feature_vector(feature_df, positive_ids)
        if pos_vec is not None:
            profile = (profile * 0.65) + (pos_vec * 0.35)

    if negative_ids:
        neg_vec = _mean_feature_vector(feature_df, negative_ids)
        if neg_vec is not None:
            profile = (profile * 0.85) - (neg_vec * 0.15)

    emotion = feedback.get("emotion_corrections", [])[-1:] or [None]
    profile = _apply_emotion_hint(profile, emotion[0])

    reason = feedback.get("skip_reasons", [])[-1:] or [None]
    profile = _apply_skip_reason(profile, reason[0])

    return profile.fillna(0.0).clip(lower=0)


def _score_candidates(
    candidate_features: pd.DataFrame, profile: pd.Series
) -> pd.Series:
    if candidate_features.empty:
        return pd.Series(dtype=float)

    feature_cols = [c for c in profile.index if c in candidate_features.columns]
    candidate_matrix = candidate_features[feature_cols].fillna(0.0).values
    profile_vec = profile[feature_cols].fillna(0.0).values.reshape(1, -1)

    if profile_vec.sum() == 0:
        # Avoid cosine_similarity complaining about zero-norm vectors.
        return pd.Series(0.0, index=candidate_features["id"])

    sims = cosine_similarity(candidate_matrix, profile_vec)[:, 0]
    return pd.Series(sims, index=candidate_features["id"])


def start_session(
    song_df: pd.DataFrame, complete_feature_set: pd.DataFrame, playlist_df: pd.DataFrame
):
    song_df = _sanitize_features(song_df)
    complete_feature_set = _sanitize_features(complete_feature_set)
    playlist_df = _sanitize_features(playlist_df)

    base_vector, nonplaylist_features = generate_playlist_feature(
        complete_feature_set, playlist_df
    )
    session_id = str(uuid.uuid4())

    SESSION_STATE[session_id] = {
        "playlist_df": playlist_df,
        "base_vector": base_vector,
        "nonplaylist_features": nonplaylist_features,
        "feedback": {
            "liked": [],
            "disliked": [],
            "emotion_corrections": [],
            "skip_reasons": [],
            "pivot_ids": [],
            "replays": [],
            "listen_events": [],
        },
        "consumed_ids": set(),
    }

    recs = get_adaptive_recommendations(
        session_id, song_df, complete_feature_set, limit=10
    )
    return session_id, recs


def apply_feedback(
    session_id: str, action: str, track_id: Optional[str], meta: Optional[Dict] = None
):
    state = SESSION_STATE.get(session_id)
    if not state:
        raise KeyError("Session not found")
    meta = meta or {}
    feedback = state["feedback"]
    consumed = state["consumed_ids"]

    if action == "like" and track_id:
        feedback["liked"].append(track_id)
        consumed.add(track_id)
    elif action == "dislike" and track_id:
        feedback["disliked"].append(track_id)
        consumed.add(track_id)
    elif action == "emotion_correction" and meta.get("emotion"):
        feedback["emotion_corrections"].append(meta["emotion"])
    elif action == "more_like_this" and track_id:
        feedback["pivot_ids"] = [track_id]
        consumed.add(track_id)
    elif action == "skip_reason" and meta.get("reason"):
        feedback["skip_reasons"].append(meta["reason"])
        if track_id:
            consumed.add(track_id)
    elif action == "replay" and track_id:
        feedback["replays"].append(track_id)
        consumed.add(track_id)
    elif action == "listen_event" and track_id:
        feedback["listen_events"].append(
            {"track_id": track_id, "seconds": meta.get("seconds", 0)}
        )
        if meta.get("seconds", 0) <= 30:
            consumed.add(track_id)


def get_adaptive_recommendations(
    session_id: str,
    song_df: pd.DataFrame,
    complete_feature_set: pd.DataFrame,
    limit: int = 5,
) -> pd.DataFrame:
    state = SESSION_STATE.get(session_id)
    if not state:
        raise KeyError("Session not found")

    profile = _build_profile(state["base_vector"], state["feedback"], complete_feature_set)

    candidate_features = state["nonplaylist_features"]
    consumed = state.get("consumed_ids", set())
    if consumed:
        candidate_features = candidate_features[~candidate_features["id"].isin(consumed)]

    scores = _score_candidates(candidate_features, profile)

    candidates = song_df[song_df["id"].isin(scores.index)].copy()
    candidates["sim"] = candidates["id"].map(scores)

    candidates = candidates.sort_values("sim", ascending=False)
    return candidates.head(limit)


def serialize_recommendations(df: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        track_id = row.get("id") or row.get("track_uri")
        artist = row.get("artist_name") or row.get("artists_song")
        title = row.get("track_name") or row.get("name")
        records.append(
            {
                "id": track_id,
                "title": title,
                "artist": artist,
                "score": round(float(row.get("sim", 0.0)), 4),
                "spotify_url": f"https://open.spotify.com/track/{track_id}"
                if track_id
                else None,
            }
        )
    return records
