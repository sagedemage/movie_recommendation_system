"""Movie Recommendation Program"""

import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from ml.dataset import MovieDataset
from ml.model import MovieRecommendation

from config import CSV_DATASET

BATCH_SIZE = 4


def main():
    if len(sys.argv) < 2:
        print("Missing the model file path!")
        exit()
    args = sys.argv
    # Use trained model
    model_path = args[1]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    data_set = MovieDataset(CSV_DATASET)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

    data_movie_ids, data_labels = next(iter(data_loader))

    # Load a saved version of the model
    saved_model = MovieRecommendation().to(device)
    saved_model.load_state_dict(torch.load(model_path, weights_only=True))

    # Perform a transform on the data for it to be usable for the model
    data_movie_ids = data_movie_ids.to(device)
    data_movie_ids = data_movie_ids.type(torch.float32)

    logits = saved_model(data_movie_ids)
    pred_probab = nn.Softplus()(logits)
    y_pred = pred_probab.argmax(0)
    pred_tensor = pred_probab[y_pred]

    movie_id = int(pred_tensor)
    title, released_year, runtime, genre, imdb_rating, director = (
        data_set.get_item_by_movie_id(movie_id)
    )

    print("Predicted movie")
    print("----------------")
    print(f"Movie ID: {movie_id}")
    print(f"Title: {title}")
    print(f"Released Year: {released_year}")
    print(f"Runtime: {runtime}")
    print(f"Genre: {genre}")
    print(f"IMDB Rating: {imdb_rating}")
    print(f"Director: {director}")
    print("")


if __name__ == "__main__":
    main()
