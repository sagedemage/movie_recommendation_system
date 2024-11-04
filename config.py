"""
Config for:
1. setting the file path for the CSV datasets
2. setting the file path for the picked movie text file
3. setting the batch size
"""

# CSV datasets
CSV_DATASET = "dataset/imdb_top_1000.csv"
CSV_TRAINING_DATASET = "training_dataset/training_movies.csv"
CSV_VALIDATION_DATASET = "validation_dataset/validation_movies.csv"
CSV_PERSONALIZED_DATASET = "personalized_dataset/recommended_movies.csv"

# The picked movie text file is used to remember what movie the user chose
PICKED_MOVIE_TEXT_FILE = "personalized_dataset/picked_movie.txt"

# The batch size for preparing data with DataLoaders
BATCH_SIZE = 5
