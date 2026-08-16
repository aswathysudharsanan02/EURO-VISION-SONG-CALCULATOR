from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Eurovision model
model = joblib.load("Eurovision_Success_Model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get inputs from the website

        song_in_english = request.form["Song.In.English"]
        group_solo = request.form["Group.Solo"]

        danceability = float(request.form["danceability"])
        energy = float(request.form["energy"])
        valence = float(request.form["valence"])
        time_signature = float(request.form["time_signature"])
        duration = float(request.form["duration"])
        happiness = float(request.form["Happiness"])
        year = float(request.form["Year"])
        acousticness = float(request.form["acousticness"])

        mode = float(request.form["mode"])
        tempo = float(request.form["tempo"])
        liveness = float(request.form["liveness"])
        key = float(request.form["key"])
        speechiness = float(request.form["speechiness"])
        loudness = float(request.form["loudness"])

        artist_gender = request.form["Artist.gender"]


        # Convert categorical values
        song_in_english = 1 if song_in_english == "Yes" else 0

        group_solo = 1 if group_solo == "Group" else 0

        if artist_gender == "Male":
            artist_gender = 0
        elif artist_gender == "Female":
            artist_gender = 1
        else:
            artist_gender = 2


        # Create input DataFrame
        input_data = pd.DataFrame([{
            "Song.In.English": song_in_english,
            "Group.Solo": group_solo,
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "time_signature": time_signature,
            "duration": duration,
            "Happiness": happiness,
            "Year": year,
            "acousticness": acousticness,
            "mode": mode,
            "tempo": tempo,
            "liveness": liveness,
            "key": key,
            "speechiness": speechiness,
            "loudness": loudness,
            "Artist.gender": artist_gender
        }])


        # Make prediction
        prediction = model.predict(input_data)[0]


        # Display prediction
        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)