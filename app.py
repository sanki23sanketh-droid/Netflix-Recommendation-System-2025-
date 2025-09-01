from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load dataset and initialize model
def initialize_model():
    df = pd.read_csv("netflix_titles.csv")
    df['combined'] = df['listed_in'].fillna('') + ' ' + df['description'].fillna('')
    df = df.dropna(subset=['title']).reset_index(drop=True)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    
    # Fit KNN
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    
    return df, knn, tfidf_matrix

df, knn, tfidf_matrix = initialize_model()

def create_similarity_plot(titles, similarities, query_title):
    plt.figure(figsize=(10, 5))
    plt.plot(titles, similarities, marker='o', linestyle='--', color='purple')
    plt.title(f"Similarity Scores for Recommendations: {query_title}")
    plt.xlabel("Recommended Titles")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode plot image to base64
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    return plot_data

def get_recommendations(title, n_recommendations=5):
    index = df[df['title'].str.lower() == title.lower()].index
    if len(index) == 0:
        return None, None, None
    
    index = index[0]
    query_title = df['title'][index]
    distances, indices = knn.kneighbors(tfidf_matrix[index], n_neighbors=n_recommendations + 1)
    
    recommendations = []
    titles = []
    similarities = []
    
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        title_rec = df['title'][i]
        similarity_score = 1 - dist  # Cosine similarity = 1 - distance
        recommendations.append({
            'title': title_rec,
            'similarity': round(similarity_score, 2),
            'type': df['type'][i],
            'description': df['description'][i],
            'genre': df['listed_in'][i]
        })
        titles.append(title_rec)
        similarities.append(similarity_score)
    
    plot_data = create_similarity_plot(titles, similarities, query_title)
    return query_title, recommendations, plot_data

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        query_title, recommendations, plot_data = get_recommendations(title)
        
        if query_title is None:
            return render_template('index.html', 
                                 error=f"'{title}' not found in our database. Please try another title.")
        
        return render_template('index.html', 
                             query_title=query_title,
                             recommendations=recommendations,
                             plot_data=plot_data)
    
    # Get random sample of titles for autocomplete suggestions
    sample_titles = df['title'].sample(10).tolist()
    return render_template('index.html', sample_titles=sample_titles)

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Please provide a title parameter'}), 400
    
    query_title, recommendations, _ = get_recommendations(title)
    
    if query_title is None:
        return jsonify({'error': f"'{title}' not found in our database"}), 404
    
    return jsonify({
        'query_title': query_title,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
