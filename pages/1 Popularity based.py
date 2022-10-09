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

st.write("### Popular Movies")


num = st.number_input("Numbers of Recommandations", value=5)

genre_1 = st.selectbox("Genres",("","Action","Adventure","Animation","Children", "Comedy", "Crime", "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","no genres listed"), key="str")

movies_3=movies
movies_3["release_year"]=movies_3["title"].str.extract("\(([0-9]+)\)")
movies_3["title"]=movies_3["title"].str.replace("\(([0-9]+)\)", "",regex=True)
movies_3["title"]=movies_3["title"].str.strip()

rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
rating['rating_count'] = ratings.groupby('movieId')['rating'].count()
rating=rating.sort_values("rating_count", ascending=False)
rating=rating.reset_index()
pop_mov=rating.merge(movies_3, how="left", on="movieId")

filter1=pop_mov["genres"].str.contains(genre_1)
pop_mov=pop_mov[["rating","title", "genres"]]
pop_mov=pop_mov[filter1].head(num)
st.table(pop_mov)
