# import all neccessary libraries

from matplotlib import image
import streamlit as st
import streamlit.components.v1 as stc
from omdbapi.movie_search import GetMovie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Define the API
movie = GetMovie(api_key='d9bd45f5')


# function to load our data
def load_data(data):
    Movies = pd.read_csv(data)
    return Movies


Movies = load_data('work_data.csv')


# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# Construct the required TF-IDF matrix by fitting and transforming the data
def vectorize_cosine(data):
    tfidf = TfidfVectorizer(stop_words='english', analyzer='word')
    tfidf_matrix = tfidf.fit_transform(data['all_word'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# function for recommendation engine


def get_recommendations(title, cosine_sim):

    # Construct a reverse map of indices and movie titles

    # Get the index of the movie that matches the title

    indices = pd.Series(Movies.index,
                        index=Movies['title']).drop_duplicates()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # we store the movie indices with their respective columns.
    movie_id = Movies['id'].iloc[movie_indices]
    movie_title = Movies['title'].iloc[movie_indices]
    movie_type = Movies['type'].iloc[movie_indices]
    movie_genres = Movies['genres'].iloc[movie_indices]
    movie_decp = Movies['description'].iloc[movie_indices]
    movie_tmdb = Movies['tmdb_popularity'].iloc[movie_indices]
    movie_year = Movies['release_year'].iloc[movie_indices]

    # We create a Pandas DataFrame with Movie_Id, Name, Genres,Production Countries, Movie Type as the columns
    recommendation_data = pd.DataFrame(
        columns=['Movie_Id', 'Name', 'Genres', 'Description', 'Popularity on TMDB', 'Movie Type', 'Movie Year'])

    recommendation_data['Movie_Id'] = movie_id
    recommendation_data['Name'] = movie_title
    recommendation_data['Genres'] = movie_genres
    recommendation_data['Description'] = movie_decp
    recommendation_data['Movie Type'] = movie_type
    recommendation_data['Popularity on TMDB'] = movie_tmdb
    recommendation_data['Release Year'] = movie_year

    return recommendation_data


# define our streamlit interface function

def main():
    # title
    st.title('Netflix Movies Recommendation Engine')
    menu = ['Recommendation', 'End page']

    section = st.sidebar.selectbox("Menu", menu)

    Movies = load_data('work_data.csv')

    if section == 'Recommendation':
        st.subheader('Recommend Movies')
        cosine_sim = vectorize_cosine(Movies)
        find_movies = st.selectbox('Search Movie', Movies['title'][1:])

        if st.button('Recommend'):
            if find_movies != None:
                res = get_recommendations(
                    find_movies, cosine_sim)

                for row in res.iterrows():
                    movie_title = row[1][1]
                    movie_type = row[1][5].title()
                    movie_Genre = row[1][2].title()
                    tmdb_rating = row[1][4]
                    movie_described = row[1][3]
                    prod_year = row[1][-1]

                    movie_ttl = movie.get_movie(title=str(movie_title))
                    col1, col2 = st.columns([1, 2])

                    try:
                        with col1:
                            st.image(movie_ttl['poster'])
                        with col2:
                            st.subheader(movie_ttl['title'])
                            st.caption(
                                f"Genre: {movie_Genre} \n ; Type: {movie_type} ")
                            st.caption(movie_described)
                            st.text(f"Tmdb Rating: {tmdb_rating}")
                            st.caption(prod_year)

                    except:
                        pass

    else:
        st.subheader('End page')
        st.image("download.jpg")

        st.caption("Thank for using my engine, Enjoy your Movie!")


if __name__ == '__main__':
    main()
