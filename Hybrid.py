import sys
sys.path.append("C:/Users/DELL/Desktop/recommend/")
import collab_filter
import content_based

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

def hybrid_recommend(user_id, movie_title, top_n=5, alpha=0.5):
    """
    alpha = weight for content-based (0.0 to 1.0)
    (1 - alpha) = weight for collaborative-based
    """

    # Step 1: Content-based similarity
    idx = indices.get(movie_title.lower())
    if idx is None:
        print("Movie not found!")
        return

    content_scores = list(enumerate(cosine_sim[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:top_n * 10]
    candidate_indices = [i[0] for i in content_scores]
    candidate_movie_ids = movies.iloc[candidate_indices]['movieId'].values

    # Step 2: Collaborative filtering predictions
    user_encoded = user2user_encoded[user_id]
    candidate_movie_encoded = [movie2movie_encoded.get(mid) for mid in candidate_movie_ids if mid in movie2movie_encoded]
    filtered_movie_ids = [mid for mid in candidate_movie_ids if mid in movie2movie_encoded]

    user_array = np.full(len(candidate_movie_encoded), user_encoded)
    movie_array = np.array(candidate_movie_encoded)

    preds = model.predict([user_array, movie_array], verbose=0).flatten()

    # Step 3: Combine both scores
    content_sim = np.array([cosine_sim[idx][indices[movies[movies['movieId'] == mid]['title'].values[0].lower()]] for mid in filtered_movie_ids])
    hybrid_scores = alpha * content_sim[:len(preds)] + (1 - alpha) * preds

    top_indices = hybrid_scores.argsort()[-top_n:][::-1]
    final_movie_ids = [filtered_movie_ids[i] for i in top_indices]

    # Step 4: Display results
    final_movies = pd.read_csv("movies.csv")
    recommended = final_movies[final_movies['movieId'].isin(final_movie_ids)][['title', 'genres']]

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Title", "Genres"],
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[recommended['title'], recommended['genres']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title=f"Top {top_n} Hybrid Recommendations for User {user_id}")
    fig.show()