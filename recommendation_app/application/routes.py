# from application import app
# from flask import Flask, render_template, request
# from application.features import *
# from application.model import *
# import os
# print(os.getcwd())    

# songDF = pd.read_csv("./data/allsong_data.csv")
# complete_feature_set = pd.read_csv("./data/complete_feature.csv")

# @app.route("/")
# def home():
#    #render the home page
#    return render_template('home.html')

# @app.route("/about")
# def about():
#    #render the about page
#    return render_template('about.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#    #requesting the URL form the HTML form
#    URL = request.form['URL']
#    #using the extract function to get a features dataframe
#    df = extract(URL)
#    #retrieve the results and get as many recommendations as the user requested
#    edm_top40 = recommend_from_playlist(songDF, complete_feature_set, df)
#    number_of_recs = int(request.form['number-of-recs'])
#    my_songs = []
#    for i in range(number_of_recs):
#       my_songs.append([(str(edm_top40.iloc[i,0]) + ' - '+ '"'+str(edm_top40.iloc[i,21])+'"', "https://open.spotify.com/track/"+ str(edm_top40.iloc[i,1]),"{:.2f}".format(edm_top40.iloc[i,-1]*100)+"%")])
#    return render_template('results.html',songs= my_songs)

from application import app
from flask import Flask, jsonify, render_template, request
from application.features import *
from application.model import *
from application.realtime import (
    apply_feedback,
    get_adaptive_recommendations,
    serialize_recommendations,
    start_session,
)
import os
from flask import request, redirect
from application.features import sp_oauth 
print(os.getcwd())    

songDF = pd.read_csv("./data/allsong_data.csv")
complete_feature_set = pd.read_csv("./data/complete_feature.csv")

@app.route("/")
def home():
   #render the home page
   return render_template('home.html')

@app.route("/about")
def about():
   #render the about page
   return render_template('about.html')

# @app.route('/callback')
# def callback():
#    """
#    This route is required for Spotify OAuth.
#    Spotipy will redirect the user here after login.
#    The OAuth token will be stored automatically in .cache.
#    """
#    return "Spotify authentication successful. You can close this window."

@app.route('/callback')
def callback():
    code = request.args.get('code')

    print(">>> CALLBACK HIT:", request.url)

    if code:
        token_info = sp_oauth.get_access_token(code)
        print("TOKEN SAVED TO CACHE:", token_info is not None)

    return "Spotify authentication successful! You can return to the app."





@app.route('/recommend', methods=['POST'])
def recommend():
   #requesting the URL form the HTML form
   ensure_token()
   URL = request.form['URL']

   #using the extract function to get a features dataframe
   df = extract(URL)

   #retrieve the results and get as many recommendations as the user requested
   edm_top40 = recommend_from_playlist(songDF, complete_feature_set, df)
   number_of_recs = int(request.form['number-of-recs'])

   my_songs = []
   for i in range(number_of_recs):
      my_songs.append([(str(edm_top40.iloc[i,0]) + ' - '+ '"'+str(edm_top40.iloc[i,21])+'"',
                        "https://open.spotify.com/track/"+ str(edm_top40.iloc[i,1]),
                        "{:.2f}".format(edm_top40.iloc[i,-1]*100)+"%")])

   return render_template('results.html', songs=my_songs)


@app.route("/api/session", methods=["POST"])
def api_start_session():
   payload = request.get_json(force=True)
   playlist_url = payload.get("playlist_url") or payload.get("URL")
   limit = int(payload.get("limit", 5))

   if not playlist_url:
      return jsonify({"error": "playlist_url is required"}), 400

   ensure_token()
   playlist_df = extract(playlist_url)
   session_id, recs = start_session(songDF, complete_feature_set, playlist_df)
   return jsonify(
      {
         "session_id": session_id,
         "recommendations": serialize_recommendations(recs.head(limit)),
      }
   )


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
   payload = request.get_json(force=True)
   session_id = payload.get("session_id")
   action = payload.get("action")
   track_id = payload.get("track_id")
   meta = payload.get("meta", {})
   limit = int(payload.get("limit", 5))

   if not session_id or not action:
      return jsonify({"error": "session_id and action are required"}), 400

   try:
      apply_feedback(session_id, action, track_id, meta)
      recs = get_adaptive_recommendations(
         session_id, songDF, complete_feature_set, limit=limit
      )
   except KeyError as exc:
      return jsonify({"error": str(exc)}), 404
   except Exception as exc:
      return jsonify({"error": str(exc)}), 400

   return jsonify(
      {"session_id": session_id, "recommendations": serialize_recommendations(recs)}
   )
