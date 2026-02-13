from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load movie data
print("Loading TMDB dataset...")
movies_df = pd.read_csv('TMDB_movie_dataset_v11.csv')
movies_df = movies_df[['title', 'overview']].dropna()
movies_df = movies_df.rename(columns={'overview': 'description'})
print(f"Loaded {len(movies_df)} movies")

# Build or load cached TF-IDF matrix
CACHE_FILE = 'tfidf_cache.pkl'

if os.path.exists(CACHE_FILE):
    print("Loading cached TF-IDF model...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    tfidf = cache['vectorizer']
    tfidf_matrix = cache['matrix']
    print("Loaded from cache!")
else:
    print("Building TF-IDF matrix (first run, will be cached)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['description'])
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'vectorizer': tfidf, 'matrix': tfidf_matrix}, f)
    print("Built and cached!")

def get_recommendations(movie_title, n=5):
    idx = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return []
    
    idx = idx[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-n-1:-1][::-1]
    
    recommendations = []
    for i in sim_indices:
        desc = movies_df.iloc[i]['description']
        recommendations.append({
            'title': movies_df.iloc[i]['title'],
            'description': desc[:150] + '...' if len(desc) > 150 else desc,
            'similarity': float(sim_scores[i])
        })
    
    return recommendations

@app.route('/')
def home():
    movie_list = movies_df['title'].tolist()[:500]  # Limit for performance
    return render_template('index.html', movies=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie')
    recommendations = get_recommendations(movie_title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
