import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

ratings=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/ratings.csv')
movies=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/movies.csv')
links=pd.read_csv('Documents/Notebooks/Recommender Systems/pages/links.csv')


st.write("### Recommandations based on Movies like")
movie_title = st.text_input('', 'Shawshank Redemption, The')
movies_1=movies
n = st.number_input("Numbers of Recommandations", value=5,key=int)
#n=math.floor(n)
#movie_title=movie_title.strip()

movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
#movies_crosstab.head(10)

def mov_rec(movtit,num):
    movies_1["release_year"]=movies_1["title"].str.extract("\(([0-9]+)\)")
    movies_1["title"]=movies_1["title"].str.replace("\(([0-9]+)\)", "",regex=True)
    movies_1["title"]=movies_1["title"].str.strip()
    movtit=movtit.strip()
    num=math.floor(num)
    klass=movies.loc[movies.title==movtit,'movieId'].values[0]
    top_movieId=klass
    topmovie_ratings = movies_crosstab[top_movieId]
    #topmovie_ratings[topmovie_ratings>=0]
    similar_to_Topmov = movies_crosstab.corrwith(topmovie_ratings)
    similar_to_Topmov.sort_values(ascending=False)
    corr_Topmov = pd.DataFrame(similar_to_Topmov, columns=['PearsonR'])
    corr_Topmov.dropna(inplace=True)
    rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    rating['rating_count'] = ratings.groupby('movieId')['rating'].count()
    corr_Topmov_summary = corr_Topmov.join(rating['rating_count'])
    top10 = corr_Topmov_summary[corr_Topmov_summary['rating_count']>=20].sort_values('PearsonR', ascending=False)
    top10=top10.reset_index()
    top10 = top10.merge(movies, how="left", on="movieId")

    top10=top10[["title","genres"]].head(num)
    #top10_gen=top10[filter1].head(n)
    return st.table(top10)
mov_rec(movie_title, n)



