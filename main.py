import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import CustomDataset
from neural_network import NeuralNetwork

BATCH_SIZE = 64

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing the model file path!")
        exit()
    args = sys.argv
    model_path = args[1]
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    csv_dataset = 'data/imdb_top_1000.csv'

    # Use trained model
    # MODEL_PATH = model_path
    # "trained_models/model_20241012_191803_0"

    training_set = CustomDataset(csv_dataset)
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

    train_movie_id, train_title, train_released_year, train_runtime, train_genre, train_imdb_rating, train_director = next(
        iter(training_loader))

    # Load a saved version of the model
    saved_model = NeuralNetwork().to(device)
    saved_model.load_state_dict(torch.load(model_path, weights_only=True))

    # Perform a transform on the data for it to be usable for the model
    train_movie_id = train_movie_id.to(device)
    train_movie_id = train_movie_id.type(torch.float32)

    print(f"Shape: {train_movie_id.shape}")
    print(f"Datatype: {train_movie_id.dtype}")
    print(f"Device: {train_movie_id.device}")
    print("")

    logits = saved_model(train_movie_id)
    pred_probab = nn.Softmax(dim=0)(logits)
    pred_tensor = train_movie_id[0].item()

    movie_id = int(pred_tensor)
    _, title, released_year, runtime, genre, imdb_rating, director = training_set.__getitem__(movie_id)

    print(f"Predicted movie")
    print(f"----------------")
    print(f"Movie ID: {movie_id}")
    print(f"Title: {title}")
    print(f"Released Year: {released_year}")
    print(f"Runtime: {runtime}")
    print(f"Genre: {genre}")
    print(f"IMDB Rating: {imdb_rating}")
    print(f"Director: {director}")
    print("")
