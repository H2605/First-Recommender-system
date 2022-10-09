import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

st.title("WBSFLX")

ratings=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/ratings.csv')
movies=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/movies.csv')
links=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/links.csv')

def transform_genre_to_regex(genres):
    regex = ""
    for genre in genres:
        regex += f"(?=.*{genre})"
    return regex

st.write("### Recommandations based on User")
uid = st.number_input("User ID",value=50,max_value=610)

num = st.number_input("Numbers of Recommandations", value=5)
genre_list = list(set([inner for outer in movies.genres.str.split('|') for inner in outer]))
genre_list.sort()
genres = st.multiselect('Optional: Select one or more genres', genre_list, default=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
genres_regex = transform_genre_to_regex(genres)

num=math.floor(num)
uid=math.floor(uid)
movies_2= movies

users_items = pd.pivot_table(data=ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')

users_items.fillna(0, inplace=True)

user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)



movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
movies_crosstab.head(10)
movies_2["release_year"]=movies_2["title"].str.extract("\(([0-9]+)\)")
movies_2["title"]=movies_2["title"].str.replace("\(([0-9]+)\)", "",regex=True)
movies_2["title"]=movies_2["title"].str.strip()

weights = (
    user_similarities.query("userId!=@uid")[uid] / sum(user_similarities.query("userId!=@uid")[uid])
          )
notwwatchedmovies = users_items.loc[users_items.index!=uid, users_items.loc[uid,:]==0]
weighted_averages = pd.DataFrame(notwwatchedmovies.T.dot(weights), columns=["predicted_rating"])
#notwwatchedmovies.T
recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
topn=recommendations.sort_values("predicted_rating", ascending=False)
filter1=topn["genres"].str.contains(genres_regex)
topn=topn[["title", "genres"]]


topn=topn[filter1].head(num)
st.table(topn)
