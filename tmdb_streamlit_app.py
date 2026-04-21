import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="TMDB Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 TMDB Movie Data Analysis & Recommendation System")
st.markdown("A simple content-based movie recommendation app built with **Streamlit**, **Pandas**, and **Scikit-learn**.")

@st.cache_data
def load_data():
    # Update this path if you place the csv somewhere else
    df = pd.read_csv("TMDB_movie_dataset_v11.csv")

    drop_cols = [
        'homepage', 'tagline', 'keywords',
        'backdrop_path', 'poster_path',
        'production_companies', 'production_countries',
        'spoken_languages', 'imdb_id'
    ]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    df = df.dropna(subset=['title', 'genres', 'release_date'])
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df = df[df['revenue'] > 0]
    df = df[df['vote_count'] > 0]
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    df['genres_clean'] = df['genres'].str.replace(',', ' ', regex=False)

    return df

@st.cache_resource
def build_similarity_matrix(genres_series: pd.Series):
    cv = CountVectorizer()
    genre_matrix = cv.fit_transform(genres_series)
    cosine_sim = cosine_similarity(genre_matrix)
    return cosine_sim


def recommend_movies(df: pd.DataFrame, cosine_sim, movie_title: str, n_recommendations: int = 5):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    cols_to_show = ['title', 'genres', 'release_year', 'vote_average', 'vote_count', 'revenue']
    existing_cols = [col for col in cols_to_show if col in df.columns]
    return df.loc[movie_indices, existing_cols]


def get_top_genres(df: pd.DataFrame, top_n: int = 10):
    genres_df = df['genres'].str.split(',', expand=True)
    genres_long = genres_df.stack().str.strip()
    return genres_long.value_counts().head(top_n)


def get_top_languages(df: pd.DataFrame, top_n: int = 10):
    return df['original_language'].value_counts().head(top_n)


def get_year_counts(df: pd.DataFrame):
    return df['release_year'].value_counts().sort_index()


df = load_data()
cosine_sim = build_similarity_matrix(df['genres_clean'])

st.sidebar.header("Controls")
section = st.sidebar.radio(
    "Choose a section",
    ["Project Overview", "EDA Dashboard", "Movie Recommender"]
)

if section == "Project Overview":
    st.subheader("Project Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Movies after cleaning", f"{len(df):,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique genres", df['genres'].str.split(',').explode().str.strip().nunique())

    st.markdown("""
### What this project does
- Cleans the TMDB movie dataset
- Analyzes genres, languages, ratings, revenue, and release trends
- Builds a content-based recommendation system using movie genres
""")

    st.subheader("Preview of Cleaned Data")
    preview_cols = ['title', 'genres', 'original_language', 'release_year', 'vote_average', 'revenue']
    existing_preview_cols = [col for col in preview_cols if col in df.columns]
    st.dataframe(df[existing_preview_cols].head(20), use_container_width=True)

elif section == "EDA Dashboard":
    st.subheader("EDA Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 Genres")
        genre_counts = get_top_genres(df)
        st.bar_chart(genre_counts)

    with col2:
        st.markdown("#### Top 10 Languages")
        lang_counts = get_top_languages(df)
        st.bar_chart(lang_counts)

    st.markdown("#### Movie Release Trend")
    year_counts = get_year_counts(df)
    st.line_chart(year_counts)

    st.markdown("#### Rating Distribution")
    rating_counts = pd.cut(df['vote_average'], bins=20).value_counts().sort_index()
    st.bar_chart(rating_counts)

    st.markdown("#### Revenue vs Rating")
    scatter_df = df[['vote_average', 'revenue']].dropna().copy()
    st.scatter_chart(scatter_df, x='vote_average', y='revenue')

elif section == "Movie Recommender":
    st.subheader("Movie Recommender")
    movie_list = sorted(df['title'].dropna().unique().tolist())
    selected_movie = st.selectbox("Choose a movie", movie_list)
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.button("Recommend"):
        results = recommend_movies(df, cosine_sim, selected_movie, top_n)
        if results.empty:
            st.error("Movie not found. Please choose another title.")
        else:
            st.success(f"Movies similar to {selected_movie}:")
            st.dataframe(results, use_container_width=True)

    st.markdown("""
### How it works
This recommendation engine uses **content-based filtering**:
1. Convert genres into text features
2. Vectorize them with `CountVectorizer`
3. Compute movie-to-movie similarity with `cosine_similarity`
4. Return the most similar titles
""")
