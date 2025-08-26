"""
Movie Semantic Search Engine
Assignment 1 - AI Systems Development

This module implements semantic search for movie plots using SentenceTransformers.
Uses the all-MiniLM-L6-v2 model to create embeddings and cosine similarity for search.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Global variables to store model and data (initialized once for efficiency)
model = None
movies_df = None
plot_embeddings = None

def load_data():
    """
    Load the movies dataset from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing movie data with columns including 'title' and 'plot'
    
    Raises:
        FileNotFoundError: If movies.csv is not found
    """
    if not os.path.exists('movies.csv'):
        raise FileNotFoundError("movies.csv not found. Please ensure the file is in the current directory.")
    
    df = pd.read_csv('movies.csv')
    return df

def initialize_model():
    """
    Initialize the SentenceTransformer model and create embeddings for all movie plots.
    This function is called automatically when needed to ensure lazy loading.
    """
    global model, movies_df, plot_embeddings
    
    if model is None:
        print("Initializing semantic search engine...")
        
        # Load the SentenceTransformer model (all-MiniLM-L6-v2 as specified)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Loaded SentenceTransformer model: all-MiniLM-L6-v2")
        
        # Load the movies dataset
        movies_df = load_data()
        print(f"✓ Loaded {len(movies_df)} movies from dataset")
        
        # Create embeddings for all movie plots
        print("Creating embeddings for all movie plots...")
        plots = movies_df['plot'].fillna('').tolist()  # Handle any NaN values
        plot_embeddings = model.encode(plots, show_progress_bar=False)
        print(f"✓ Created embeddings with shape: {plot_embeddings.shape}")
        
        print("Search engine initialized successfully!\n")

def search_movies(query, top_n=5):
    """
    Search for movies based on semantic similarity to the query.
    
    This function uses SentenceTransformers to encode the query and finds
    the most similar movie plots using cosine similarity.
    
    Args:
        query (str): The search query describing desired movie characteristics
        top_n (int, optional): Number of top results to return. Defaults to 5.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['title', 'plot', 'similarity']
                     sorted by similarity score in descending order
    
    Example:
        >>> results = search_movies('spy thriller in Paris', top_n=3)
        >>> print(results.head())
    """
    # Initialize the model and data if not already done
    initialize_model()
    
    # Validate inputs
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    # Ensure top_n doesn't exceed available movies
    top_n = min(top_n, len(movies_df))
    
    # Encode the search query using the same model
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity between query and all movie plot embeddings
    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
    
    # Get indices of top_n most similar movies (sorted in descending order)
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Create result DataFrame with the most similar movies
    results = movies_df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    # Return DataFrame with required columns, reset index for clean output
    return results[['title', 'plot', 'similarity']].reset_index(drop=True)

def main():
    """
    Main function to demonstrate the movie search functionality.
    Tests the search with the required query: 'spy thriller in Paris'
    """
    print("Movie Semantic Search Engine - Assignment 1")
    print("=" * 50)
    
    try:
        # Test with the specific query mentioned in assignment
        test_query = 'spy thriller in Paris'
        print(f"Searching for: '{test_query}'")
        print("-" * 50)
        
        # Perform the search
        results = search_movies(test_query, top_n=5)
        
        print(f"Top {len(results)} movies matching '{test_query}':\n")
        
        # Display results in a formatted way
        for i, row in results.iterrows():
            print(f"{i+1}. {row['title']}")
            print(f"   Similarity Score: {row['similarity']:.4f}")
            print(f"   Plot: {row['plot'][:150]}{'...' if len(row['plot']) > 150 else ''}")
            print()
        
        # Additional test queries for demonstration
        additional_queries = [
            'romantic comedy in New York',
            'space adventure with aliens',
            'horror movie in a haunted house'
        ]
        
        print("\nAdditional test searches:")
        print("-" * 30)
        
        for query in additional_queries:
            results = search_movies(query, top_n=3)
            print(f"\n'{query}' - Top match: {results.iloc[0]['title']} (Score: {results.iloc[0]['similarity']:.4f})")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure movies.csv is in the current directory and contains 'title' and 'plot' columns.")

if __name__ == "__main__":
    main()