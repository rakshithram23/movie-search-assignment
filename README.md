# Movie Semantic Search Assignment

This repository contains my solution for the semantic search on movie plots assignment using SentenceTransformers.

## Overview

The project implements a semantic search engine that can find movies based on natural language queries about their plots. It uses the `all-MiniLM-L6-v2` SentenceTransformer model to create embeddings and cosine similarity to find the most relevant movies.

## Features

- **Semantic Search**: Find movies using natural language descriptions
- **Efficient Implementation**: Lazy loading of models and embeddings
- **Robust Error Handling**: Validates inputs and handles edge cases
- **Comprehensive Testing**: Full unit test coverage with 4 test cases
- **Easy to Use**: Simple function interface with clear documentation

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
   
   git clone https://github.com/your-username/movie-search-assignment.git
   cd movie-search-assignment
   

2. **Create and activate virtual environment:**

   # Create virtual environment
   python -m venv venv
   
   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   

3. **Install dependencies:**

   pip install -r requirements.txt
   

4. **Dataset:**
   - The `movies.csv` file should be in the root directory
   - For testing, a sample dataset with 3 movies is used

## Usage

### Basic Usage


from movie_search import search_movies

# Search for movies
results = search_movies('spy thriller in Paris', top_n=5)
print(results)


### Run the Demo


python movie_search.py


This will run a demonstration with several example queries including the required test case: `'spy thriller in Paris'`.

### Example Output


Movie Semantic Search Engine - Assignment 1
==================================================
Searching for: 'spy thriller in Paris'
--------------------------------------------------
Initializing semantic search engine...
✓ Loaded SentenceTransformer model: all-MiniLM-L6-v2
✓ Loaded 3 movies from dataset
Creating embeddings for all movie plots...
✓ Created embeddings with shape: (3, 384)
Search engine initialized successfully!

Top 3 movies matching 'spy thriller in Paris':

1. Spy Movie
   Similarity Score: 0.7697
   Plot: A spy navigates intrigue in Paris to stop a terrorist plot.

2. Romance in Paris
   Similarity Score: 0.3880
   Plot: A couple falls in love in Paris under romantic circumstances.

3. Action Flick
   Similarity Score: 0.2568
   Plot: A high-octane chase through New York with explosions.


## Testing

The project includes comprehensive unit tests that verify:

1. **Output Format**: Correct DataFrame structure with required columns (`title`, `plot`, `similarity`)
2. **Top-N Parameter**: Returns exactly `top_n` number of results
3. **Similarity Range**: Similarity scores are between 0 and 1
4. **Query Relevance**: Search results are semantically relevant to the input query

### Run Tests


# Run all tests with verbose output
python -m unittest tests/test_movie_search.py -v


### Expected Test Output


test_search_movies_output_format (tests.test_movie_search.TestMovieSearch) ... ok
test_search_movies_relevance (tests.test_movie_search.TestMovieSearch) ... ok  
test_search_movies_similarity_range (tests.test_movie_search.TestMovieSearch) ... ok
test_search_movies_top_n (tests.test_movie_search.TestMovieSearch) ... ok

----------------------------------------------------------------------
Ran 4 tests in 8.050s

OK


## Project Structure


movie-search-assignment/
├── movie_search.py           # Main implementation
├── tests/
│   └── test_movie_search.py  # Unit tests
├── movies.csv               # Movie dataset
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules

## Implementation Details

### Core Function

The main function `search_movies(query, top_n=5)` performs the following steps:

1. **Initialization**: Loads the SentenceTransformer model and creates embeddings for all movie plots (lazy loading)
2. **Query Encoding**: Converts the search query into a numerical embedding using the same model
3. **Similarity Calculation**: Computes cosine similarity between query and all movie plot embeddings
4. **Ranking**: Sorts movies by similarity score in descending order
5. **Results**: Returns top N movies as a pandas DataFrame with columns: `title`, `plot`, `similarity`

### Key Features

- **Lazy Loading**: Model and embeddings are initialized only when first needed for efficiency
- **Error Handling**: Validates inputs and provides meaningful error messages
- **Performance**: Efficient numpy operations for similarity calculations
- **Flexibility**: Configurable number of results via `top_n` parameter
- **Robustness**: Handles NaN values in plots and ensures top_n doesn't exceed available movies

## Dependencies
sentence-transformers
pandas
scikit-learn
numpy
torch


## Assignment Requirements Met

✅ **Install and import libraries**: All required libraries properly imported and used  
✅ **Load movies.csv**: Dataset loaded into pandas DataFrame with error handling  
✅ **Create embeddings**: Uses all-MiniLM-L6-v2 model as specified  
✅ **Implement search_movies()**: Function returns DataFrame with correct columns (`title`, `plot`, `similarity`) and top_n results  
✅ **Test with required query**: Successfully tested with 'spy thriller in Paris'  
✅ **Unit tests**: All 4 tests pass with proper validation  
✅ **Code quality**: Clean, commented, and well-documented code with proper error handling  
✅ **Documentation**: Comprehensive README with setup and usage instructions  

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies are installed
2. **FileNotFoundError**: Make sure `movies.csv` is in the project root directory
3. **Memory Issues**: The model download and embedding creation require sufficient RAM
4. **Slow First Run**: Initial model download and embedding creation takes time

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.9+)
3. Ensure movies.csv is present and readable
4. Check the error message for specific guidance

## Test Results

All unit tests pass successfully:
- **Output format test**: ✅ Returns DataFrame with correct columns
- **Top-N parameter test**: ✅ Returns exactly the requested number of results  
- **Similarity range test**: ✅ All similarity scores are between 0 and 1
- **Query relevance test**: ✅ Results are semantically relevant to the search query

## Author

Rama Rakshith Katta(221020432) - AI Systems Development Course, IIIT Naya Raipur

## License

This project is for educational purposes as part of the AI Systems Development course assignment.