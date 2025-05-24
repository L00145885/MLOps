import pickle

from flask import Flask, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
