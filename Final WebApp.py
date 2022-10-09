import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

st.title("WBSFLX")

st.write("""
### Project description
As a freelance Data Scientist, a customer approaches you with an ambitious request: she wants to take her DVD store online. And you thought all DVD stores were dead! Not quite: her store, called WBSFLIX, operates in a small town near Berlin and is still alive thanks to a loyal customer base that appreciates the local atmosphere and, more than anything, the personal recommendations of the owner, Ursula.
 
""")


ratings=pd.read_csv('Documents/Notebooks/Recommender Systems/ml-latest-small/ratings.csv')
movies=pd.read_csv('Documents/Notebooks/Recommender Systems/ml-latest-small/movies.csv')
links=pd.read_csv('Documents/Notebooks/Recommender Systems/ml-latest-small/links.csv')

st.write("### Popular Movies")

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
pop_mov=pop_mov[filter1].head(10)
st.table(pop_mov)


st.write("### Recommandations based on Movies like")
movie_title = st.text_input('', 'Shawshank Redemption, The')
movies_1=movies
n = st.number_input("Numbers of Recommandations", value=5,key=int)
n=math.floor(n)

movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
#movies_crosstab.head(10)
movies_1["release_year"]=movies_1["title"].str.extract("\(([0-9]+)\)")
movies_1["title"]=movies_1["title"].str.replace("\(([0-9]+)\)", "",regex=True)
movies_1["title"]=movies_1["title"].str.strip()
klass=movies.loc[movies.title==movie_title,'movieId'].values[0]
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

top10=top10[["title","genres"]].head(n)
#top10_gen=top10[filter1].head(n)
st.table(top10)




st.write("### Recommandations based on User")
uid = st.number_input("User ID",value=50,max_value=610)
#uid = st.number_input("Enter User Id", value=50,max_value=610)
#st.write("### Recommandations based on User ",uid) 
num = st.number_input("Numbers of Recommandations", value=5)
genre = st.selectbox("",("","Action","Adventure","Animation","Children", "Comedy", "Crime", "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","no genres listed"))

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
filter1=topn["genres"].str.contains(genre)
topn=topn[["title", "genres"]]


topn=topn[filter1].head(num)
st.table(topn)
