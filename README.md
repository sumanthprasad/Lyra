## Lyra: Emotion-Aware, Feedback-Driven Music Recommender

### Authors
- Sumanth Sai Prasad  
- Ramyashree Mummuderlapalli Krishnaiah

Lyra generates personalized song recommendations from any Spotify playlist. It blends lyric emotion, audio features, and your real-time feedback to keep the queue on your vibe.

- **Stack:** Flask API, PostgreSQL-ready, Spotipy, scikit-learn, TextBlob, GAN-based recommender, Bootstrap/custom UI.
- **Signals:** Audio features (tempo, energy, valence, danceability, acousticness, loudness, speechiness), 13-category lyric emotion labels, cosine similarity for re-ranking.
- **Adaptive loop:** Like/Dislike, Pivot closer, Skip reasons, Replays/fast skips, Mood corrections all update the session profile instantly.


### Quickstart
```sh
git clone https://github.com/sumanthprasad/Lyra.git
cd Novel_Song_Recommender-main
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r recommendation_app/requirements.txt
cd recommendation_app
python wsgi.py
# visit http://127.0.0.1:8000
```

### Using the adaptive UI
1) Paste a Spotify playlist URL on the home page and start a session.  
2) Interact with the cards:  
   - 👍 Mood match / 👎 Not it  
   - 🔄 Pivot closer (re-rank around that track)  
   - ⏭️ Skip reason (wrong energy/mood/genre/don't like)  
   - 😊 Played through / ⏭️ Skipped fast / 🔁 Replayed  
   - 😊 Mood correction dropdown (happy, sad, calm, angry, excited, romantic, nostalgic)  
3) Watch the queue refresh instantly after each click.

### Data collection
- Seeded from Spotify playlist: [link](https://open.spotify.com/playlist/1G8IpkZKobrIlXcVPoSIuf?si=f11fb54e99334cd9) (~10k songs).
- Spotipy-based harvesting in `notebooks/extraction_api.ipynb`; update the playlist ID and provide your own `client_id`/`client_secret` (see [Spotify app settings](https://developer.spotify.com/documentation/general/guides/authorization/app-settings/)).
- Precomputed feature CSVs live in `data/` (and `recommendation_app/data1/`).

### Architecture highlights
- **Backend:** Flask routes + JSON APIs for sessions (`/api/session`) and feedback (`/api/feedback`); legacy `/recommend` kept for SSR results.
- **Realtime core:** `recommendation_app/application/realtime.py` tracks session state, builds profiles from feedback, sanitizes feature vectors, and re-ranks via cosine similarity.
- **Models:** Playlist feature summarization + GAN-based recommender with cosine re-ranking; emotion hints and skip-reason biases steer valence/energy/danceability/etc.
- **Frontend:** Single-page adaptive UI (Bootstrap + custom CSS) with live fetch calls and card-based controls.

### Deployment notes
- Default redirect URI: `http://127.0.0.1:8000/callback` (see `application/features.py`).
- Requires local `.cache` for Spotify OAuth; login once in browser when prompted.
- PostgreSQL is optional; current app runs with CSV-backed features.

### Links 
- Notebooks: [Data extraction](https://github.com/shivam360d/Novel_Song_Recommender/blob/main/notebooks/extraction_api.ipynb) | [EDA & clustering](https://github.com/shivam360d/Novel_Song_Recommender/blob/main/notebooks/cluster_analysis.ipynb)  


