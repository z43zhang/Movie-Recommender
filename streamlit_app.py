import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
from dotenv import load_dotenv
import os

import requests

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        path = data.get("poster_path", None)
        if path:
            return f"https://image.tmdb.org/t/p/w200{path}"
    return "https://via.placeholder.com/200x300?text=No+Image"


st.markdown(
    """
    <style>
    /* Center align table headers and cells */
    thead tr th {
        text-align: center !important;
    }

    tbody tr td {
        text-align: center !important;
    }

    /* Optional: Center the whole table */
    .dataframe {
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")

    def combine_genres_keywords(row):
        genres = ' '.join(''.join(g['name'].split()) for g in json.loads(row['genres']))
        keywords = ' '.join(''.join(k['name'].split()) for k in json.loads(row['keywords']))
        return f"{genres} {keywords}"

    def build_embedding_text(row):
        genres = ' '.join(''.join(g['name'].split()) for g in json.loads(row['genres']))
        keywords = ' '.join(''.join(k['name'].split()) for k in json.loads(row['keywords']))
        overview = row.get('overview', '')
        return f"{genres} {keywords} {overview}"

    df['tfidf_text'] = df.apply(combine_genres_keywords, axis=1)
    df['embedding_text'] = df.apply(build_embedding_text, axis=1)
    return df

df = load_data()

# TF-IDF + SVD
@st.cache_resource
def load_tfidf_models():
    vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    matrix = np.load("artifacts/tfidf_matrix.npy")
    svd = joblib.load("artifacts/tfidf_svd.pkl")
    reduced = np.load("artifacts/tfidf_reduced.npy")
    return vectorizer, matrix, svd, reduced

tfidf_vectorizer, tfidf_matrix, svd, tfidf_reduced = load_tfidf_models()

tfidf_matrix = tfidf_vectorizer.fit_transform(df['tfidf_text'])

# Cache Model Loading
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


@st.cache_data
def get_embeddings(text_list):
    return embedding_model.encode(text_list, show_progress_bar=True)

# embedding_matrix = get_embeddings(df['embedding_text'].tolist())
# load precomputed embedding matrix for fast response
embedding_matrix = np.load("artifacts/embedding_matrix.npy")


# Movie title → index
title_to_index = pd.Series(df.index, index=df['title'])

# Title matching
def match_title(input_title, titles, threshold=80):
    match, score = process.extractOne(input_title, titles)
    return match if score >= threshold else None

# Recommendation functions
def recommend_tfidf(input_title, top_n=10):
    matched_title = match_title(input_title, title_to_index.index)
    if not matched_title:
        return f"Movie '{input_title}' not found."

    idx = title_to_index[matched_title]
    query_vector = tfidf_reduced[idx].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vector, tfidf_reduced).flatten()
    top_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    return [(df.iloc[i]['title'], similarity_scores[i]) for i in top_indices]

def recommend_embedding(input_title, top_n=10):
    matched_title = match_title(input_title, title_to_index.index)
    if not matched_title:
        return f"Movie '{input_title}' not found."

    idx = title_to_index[matched_title]
    query_vector = embedding_matrix[idx].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vector, embedding_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    return [(df.iloc[i]['title'], similarity_scores[i]) for i in top_indices]

def hybrid_search(query, alpha=0.8, top_n=10):
    query_embed = embedding_model.encode([query])
    query_tfidf = tfidf_vectorizer.transform([query])

    sim_embed = cosine_similarity(query_embed, embedding_matrix).flatten()
    sim_tfidf = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    combined_score = alpha * sim_embed + (1 - alpha) * sim_tfidf
    top_indices = combined_score.argsort()[::-1][:top_n]
    return [(df.iloc[i]['title'], combined_score[i]) for i in top_indices]

# --- Streamlit UI ---

st.markdown("<h1 style='text-align: center;'>🎬Movie Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: right; color: gray; margin-top: -10px; margin-bottom: 30px;">
    Developed by Zhonghao Zhang
</div>
""", unsafe_allow_html=True)


query = st.text_input("Enter a movie title or description", "I want to watch Marvel movies")

method = st.selectbox("Select Recommendation Method", [
    "TF-IDF", "Embedding", "Hybrid"
])

st.markdown("""
<style>
div.stButton > button {
    background-color: #6c63ff;
    color: white;
    font-weight: 500;
    border-radius: 8px;
    padding: 0.5em 2em;
}
div.stButton > button:hover {
    background-color: #5548c8;
}
</style>
""", unsafe_allow_html=True)


# Center the button using Streamlit layout properly
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    recommend_clicked = st.button("Recommend", use_container_width=True)

if recommend_clicked:
    with st.spinner("Fetching results..."):
        if method == "TF-IDF":
            results = recommend_tfidf(query)
        elif method == "Embedding":
            results = recommend_embedding(query)
        else:
            results = hybrid_search(query)

    if isinstance(results, str):
        st.warning(results)
    else:
        st.markdown("### 🎥 Top Recommendations")

        for title, score in results:
            row = df[df['title'] == title].iloc[0]
            movie_id = row['id']
            poster_url = fetch_poster(movie_id)
            tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}"

            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="{poster_url}" width="100" style="border-radius: 8px; margin-right: 20px;">
                <div>
                    <h4 style="margin: 0;">{title}</h4>
                    <p style="margin: 0;">Score: {score:.4f}</p>
                    <a href="{tmdb_url}" target="_blank">🔗 View on TMDB</a>
                </div>
            </div>
            """, unsafe_allow_html=True)


