import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import recom


app = Flask(__name__)


@app.route("/")
def home():

    return render_template('index.html')


@app.route('/movie', methods=['POST'])
def recommend_movies():
    title = request.form['the_title']

    result_final = recom.get_recommendations(title)

    name = []
    Genres = []
    Production_Countries = []
    Movie_Type = []
    for i in range(len(result_final)):
        name.append(result_final.iloc[i][0])
        Genres.append(result_final.iloc[i][1])
        Production_Countries.append(result_final.iloc[i][2])
        Movie_Type.append(result_final.iloc[i][3])

    return render_template('index.html', movie_name=name, movie_genre=Genres,
                           movie_county=Production_Countries, movie_type=Movie_Type,  search_name=title)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
