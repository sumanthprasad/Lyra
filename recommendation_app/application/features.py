print("LOADED FEATURES.PY:", __file__)

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import re
import time

CLIENT_ID = "6d0cfc8d9b914b08b81c3a92d2ba2cd6"
CLIENT_SECRET = "662785711d244923899501f03b349e48"
REDIRECT_URI = "http://127.0.0.1:8000/callback"
SCOPE = "playlist-read-private playlist-read-collaborative user-library-read user-read-private user-read-playback-state"


sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".cache",
    show_dialog=True,
)

sp = spotipy.Spotify(auth_manager=sp_oauth)


# -----------------------------------------------------
# TOKEN
# -----------------------------------------------------
def ensure_token():
    token_info = sp_oauth.get_cached_token()
    if token_info:
        return token_info["access_token"]

    auth_url = sp_oauth.get_authorize_url()
    print("\nLOGIN HERE:", auth_url)

    import webbrowser
    webbrowser.open(auth_url)

    for _ in range(60):
        token_info = sp_oauth.get_cached_token()
        if token_info:
            print("TOKEN RECEIVED!")
            return token_info["access_token"]
        time.sleep(1)

    raise Exception("Spotify login timeout.")


# -----------------------------------------------------
# RAW PLAYLIST FETCH
# -----------------------------------------------------
def get_playlist_items_raw(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=100"
    headers = {"Authorization": f"Bearer {token}"}

    all_items = []
    next_url = url

    while next_url:
        r = requests.get(next_url, headers=headers)

        if r.status_code != 200:
            raise Exception(f"Playlist fetch failed: {r.status_code} -> {r.text}")

        data = r.json()
        all_items.extend(data["items"])
        next_url = data["next"]

    return all_items


# -----------------------------------------------------
# EXTRACT FUNCTION
# -----------------------------------------------------
def extract(URL):
    token = ensure_token()

    match = re.search(r"playlist/([A-Za-z0-9]+)", URL)
    if not match:
        raise Exception("Invalid playlist URL")

    playlist_id = match.group(1)
    print("Playlist ID =", playlist_id)

    items = get_playlist_items_raw(token, playlist_id)

    track_ids = []
    track_names = []
    track_artists = []
    track_first_artists = []

    # ----------- PARSE TRACKS -------------
    for item in items:
        t = item.get("track")
        if not t or not t.get("id"):
            continue

        track_ids.append(t["id"])
        track_names.append(t["name"])
        artists = [a["name"] for a in t["artists"]]
        track_artists.append(artists)
        track_first_artists.append(artists[0])

    # ----------- WORKAROUND: Use existing data instead of API -------------
    print(f"\n⚠️  Using pre-computed features (Spotify API audio-features blocked)")
    print(f"   Found {len(track_ids)} tracks in playlist\n")
    
    # Load the existing song database
    try:
        complete_feature_set = pd.read_csv("./data/complete_feature.csv")
    except:
        raise Exception("Could not load complete_feature.csv - make sure it exists in ./data/")
    
    # Filter to only tracks that are in our database
    df = complete_feature_set[complete_feature_set['id'].isin(track_ids)]
    
    if len(df) == 0:
        raise Exception(f"None of the {len(track_ids)} tracks in this playlist are in our database. Try a playlist with more popular songs.")
    
    print(f"✓ Matched {len(df)} out of {len(track_ids)} tracks from our database")
    
    # Add metadata for matched tracks
    matched_ids = df['id'].tolist()
    matched_names = [track_names[i] for i, tid in enumerate(track_ids) if tid in matched_ids]
    matched_artists = [track_first_artists[i] for i, tid in enumerate(track_ids) if tid in matched_ids]
    
    df = df.copy()
    df['title'] = matched_names[:len(df)]
    df['first_artist'] = matched_artists[:len(df)]
    
    expected_cols = [
        "id","danceability","energy","key","loudness","mode",
        "acousticness","instrumentalness","liveness","valence",
        "tempo","duration_ms","time_signature"
    ]

    keep_cols = [c for c in expected_cols if c in df.columns]
    
    return df[["title","first_artist"] + keep_cols]