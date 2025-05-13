import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    return pd.read_pickle(r"C:/Users/prati/Desktop/Spotify Music Recommender/data/preprocessed/df.pkl")

@st.cache_data
def load_latent_features():
    return np.load(r"C:/Users/prati/Desktop/Spotify Music Recommender/models/trained/latent_features.npy")

@st.cache_resource
def load_knn_model():
    return joblib.load(r"C:/Users/prati/Desktop/Spotify Music Recommender/models/trained/knn_model.pkl")

@st.cache_data
def calculate_pca(latent_features, n_components=3):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(latent_features)
    return components

@st.cache_data
def calculate_correlation(df, numeric_features):
    return df[numeric_features].corr()


numeric_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'track_genre_encoded'
]

st.set_page_config(page_title="Spotify Recommender", layout="wide")
st.title("🎧 Music Recommender")

tab1, tab2 = st.tabs(["🎵 Recommendations", "📊 Genre Visualization", ])

def plot_heatmap(df, numeric_features):
    corr = calculate_correlation(df, numeric_features)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=numeric_features,
        y=numeric_features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}'
    ))
    fig.update_layout(title='Feature Correlation Heatmap', width=700, height=700)
    return fig

def plot_latent_3d(latent_features, df):
    components = calculate_pca(latent_features)
    plot_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    plot_df['Genre'] = df['track_genre']
    plot_df['Track'] = df['track_name']
    plot_df['Artist'] = df['artists']
    fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3',
                        color='Genre', hover_data=['Track', 'Artist'],
                        title="3D Genre Visualization")
    return fig

def recommend_tracks(track_index, df, latent_features, knn):
    track_vector = latent_features[track_index].reshape(1, -1)
    distances, indices = knn.kneighbors(track_vector, n_neighbors=30)
    similar_indices = indices.flatten()[1:]
    similar_tracks = df.iloc[similar_indices]
    unique_tracks = similar_tracks.drop_duplicates(subset=['track_name', 'artists']).head(9)
    return unique_tracks[['track_name', 'artists', 'track_genre']]

with tab1:
    st.subheader("Get Track Recommendations")
    df = load_data().reset_index(drop=True)
    latent_features = load_latent_features()
    knn = load_knn_model()

    track_options = [f"{i} - {row['track_name']} by {row['artists']}" for i, row in df.iterrows()]
    selected_option = st.selectbox("Select a Track:", track_options)
    track_index = int(selected_option.split(" - ")[0])

    selected = df.iloc[track_index]
    st.markdown(f"**Selected:** {selected['track_name']} by {selected['artists']} | Genre: *{selected['track_genre']}*")

    if 'track_id' in df.columns:
        track_id = selected['track_id']
        components.html(
            f"""
            <iframe src="https://open.spotify.com/embed/track/{track_id}"
                    width="100%" height="80" frameborder="0"
                    allowtransparency="true" allow="encrypted-media">
            </iframe>
            """,
            height=100
        )
    else:
        st.warning("Track ID not available - can't play preview")

    st.write("**Top Recommendations:**")
    recs = recommend_tracks(track_index, df, latent_features, knn)

    st.markdown("""
    <style>
        .dark-card {
            background-color: #000000 !important;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white !important; /* Default text color to white */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .dark-card iframe {
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'track_id' in df.columns:
        cols = st.columns(3)  # Display in 3 columns for better use of space
        for i, (idx, row) in enumerate(recs.iterrows()):
            original_track = df[(df['track_name'] == row['track_name']) &
                                (df['artists'] == row['artists'])].iloc[0]
            with cols[i % 3]:
                components.html(
                    f"""
                    <div class="dark-card">
                        <iframe src="https://open.spotify.com/embed/track/{original_track['track_id']}"
                                width="100%" height="80" frameborder="0"
                                allowtransparency="true" allow="encrypted-media">
                        </iframe>
                    </div>
                    """,
                    height=100  # Adjust height as needed
                )
    else:
        st.dataframe(recs)

with tab2:
    st.subheader("3D PCA Visualization of Tracks by Genre")
    df_viz = load_data()
    latent_features_viz = load_latent_features()
    fig = plot_latent_3d(latent_features_viz, df_viz)
    st.plotly_chart(fig, use_container_width=True)