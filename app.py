from nltk import word_tokenize
from string import punctuation
from flask import *
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB




app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            inp = request.form["inp"] 
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(inp)
            if score["compound"] >= 0.05:
                return render_template("home.html", msg="Positive 😊😀")
            elif score["compound"] <= -0.05:
                return render_template("home.html", msg = "Negative 😔😞")
            else:
                return render_template("home.html", msg = "Neutral 😐")
        except Exception as e:
            msg = "Issue " + str(e)
            return render_template("home.html", msg = msg)
    return render_template("home.html")

    


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
