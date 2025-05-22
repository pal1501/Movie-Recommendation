import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

# Load data (MovieLens)
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Encode user and movie IDs
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user2user_encoded)
ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)
num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# Build training set
X = ratings[['user', 'movie']].values
y = ratings['rating'].values

# User input and embedding
user_input = layers.Input(shape=(1,))
user_embedding = layers.Embedding(num_users, 64)(user_input)
user_vec = layers.Flatten()(user_embedding)

# Movie input and embedding
movie_input = layers.Input(shape=(1,))
movie_embedding = layers.Embedding(num_movies, 64)(movie_input)
movie_vec = layers.Flatten()(movie_embedding)

# Concatenate and dense layers
concat = layers.Concatenate()([user_vec, movie_vec])
dense = layers.Dense(128, activation='relu')(concat)
dropout = layers.Dropout(0.5)(dense)
dense = layers.Dense(64, activation='relu')(dropout)
output = layers.Dense(1)(dense)

# Compile the model
model = tf.keras.Model([user_input, movie_input], output)
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(
    x=[X[:, 0], X[:, 1]],
    y=y,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

def recommend_movies(user_id, n=10):
    # Encode the user
    user_encoded = user2user_encoded[user_id]
    watched_movies = ratings[ratings['userId'] == user_id]['movieId'].map(movie2movie_encoded)

    # Find unwatched movies
    all_movie_indices = np.array(list(movie2movie_encoded.values()))
    unwatched = np.setdiff1d(all_movie_indices, watched_movies)

    # Predict ratings
    user_array = np.full_like(unwatched, fill_value=user_encoded)
    preds = model.predict([user_array, unwatched], verbose=0)

    # Get top-N recommendations
    top_indices = preds.flatten().argsort()[-n:][::-1]
    recommended_movie_ids = [movie_ids[i] for i in unwatched[top_indices]]

    # Fetch movie details
    movie_df = pd.read_csv('movies.csv')
    recommended = movie_df[movie_df['movieId'].isin(recommended_movie_ids)][['title', 'genres']]

    # Plot as table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Title", "Genres"],
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[recommended['title'], recommended['genres']],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title=f"Top {n} Movie Recommendations for User {user_id}")
    # Save and open in browser
    fig.write_html("test_plot.html")
    fig.show()

recommend_movies(user_id=1, n=10)

model.save('movie_rec_model.h5')

