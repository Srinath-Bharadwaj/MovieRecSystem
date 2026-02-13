# Movie Recommendation System

Content-based movie recommendation system using TF-IDF vectorization and cosine similarity.

## Features
- TF-IDF vectorization for text analysis
- Cosine similarity for finding similar movies
- Flask web interface
- Fast response time with optimized algorithm

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure `TMDB_movie_dataset_v11.csv` is in the project directory

3. Run the app:
```bash
python app.py
```

4. Open browser to `http://127.0.0.1:5000`

## Usage
1. Type or select a movie title from the dropdown
2. Click "Get Recommendations"
3. View similar movies with similarity scores
