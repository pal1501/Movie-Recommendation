import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('C:/Users/DELL/Desktop/recommend/movies.csv')
tags = pd.read_csv('C:/Users/DELL/Desktop/recommend/tags.csv')
print(movies.head())

# Convert 'Action|Adventure|Sci-Fi' → 'Action Adventure Sci-Fi'
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Ensure clean column names
movies.columns = movies.columns.str.strip()
tags.columns = tags.columns.str.strip()
print(movies.columns)
print(tags.columns)

# Merge tags into one string per movie
tag_data = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x.astype(str))).reset_index()
print(tag_data.head())  # Check that movieId is present

# Merge tags with movies
if 'tag' in movies.columns:
    movies = movies.drop(columns=['tag'])

movies = movies.merge(tag_data, on='movieId', how='left')
print(movies.columns.tolist())
movies.head()

# Combine genres and tags as one text blob per movie
movies['combined_features'] = movies['genres'].fillna('') + " " + movies['tag'].fillna('')
print(movies.head())

# Use TF-IDF to vectorize text
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Build a reverse index for movie titles
movies['normalized_title'] = movies['title'].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies['normalized_title']).drop_duplicates()
print(indices)

def recommend_movies(title, top_n=5):
    title = title.lower()
    if title not in indices:
        print("Movie not found!")
        return pd.DataFrame()  # Return empty DataFrame if not found
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    # Return as DataFrame
    return movies.iloc[movie_indices][['title', 'genres']].reset_index(drop=True)
recommendations = recommend_movies('toy Story (1995)', top_n=10)
print(recommendations) 
print(recommendations.to_markdown())  