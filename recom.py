import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

Movies_credits_new = pd.read_csv('Movies.csv')

tfidf = TfidfVectorizer(stop_words='english')

Movies_credits_new['combine'] = Movies_credits_new['description'] + " " + \
    Movies_credits_new['production_countries'] + \
    " " + Movies_credits_new['genres']
tfidf_matrix = tfidf.fit_transform(Movies_credits_new['combine'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(Movies_credits_new.index,
                    index=Movies_credits_new['title']).drop_duplicates()


def get_recommendations(m_name, cosine_sim=cosine_sim):

    # Construct a reverse map of indices and movie titles
    # Get the index of the movie that matches the title
    idx = indices[m_name]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # we store the movie indices with their respective columns.
    # movie_id = Movies_credits_new['id'].iloc[movie_indices]
    movie_title = Movies_credits_new['title'].iloc[movie_indices]
    movie_type = Movies_credits_new['type'].iloc[movie_indices]
    movie_genres = Movies_credits_new['genres'].iloc[movie_indices]
    movie_country = Movies_credits_new['production_countries'].iloc[movie_indices]

    # We create a Pandas DataFrame with Movie_Id, Name, Genres,Production Countries, Movie Type as the columns
    recommendation_data = pd.DataFrame(
        columns=['Name', 'Genres', 'Production Countries', 'Movie Type'])

    # recommendation_data['Movie_Id'] = movie_id
    recommendation_data['Name'] = movie_title
    recommendation_data['Genres'] = movie_genres
    recommendation_data['Production Countries'] = movie_country
    recommendation_data['Movie Type'] = movie_type

    return recommendation_data


def results(movie_name):
    movie_name = movie_name.lower()

    find_movie = Movies_credits_new['title']

    if movie_name not in find_movie['title'].unique():
        return 'Movie not in Database'

    else:
        recommendations = get_recommendations(movie_name)
        return recommendations.to_dict('records')
