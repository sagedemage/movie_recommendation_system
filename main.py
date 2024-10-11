import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

from dataset import CustomDataset
from neural_network import NeuralNetwork

csv_dataset_file = 'data/imdb_top_1000.csv'

dataset = CustomDataset(csv_dataset_file)

print("Length: " + str(dataset.__len__()))
print("")

movie_id, title, released_year, runtime, genre, imdb_rating, director = dataset.__getitem__(0)
print(f"Movie ID: {movie_id}")
print(f"Title: {title}")
print(f"Released Year: {released_year}")
print(f"Runtime: {runtime}")
print(f"Genre: {genre}")
print(f"IMDB Rating: {imdb_rating}")
print(f"Director: {director}")
print("")

# batch_size (int, optional) – how many samples per batch to load (default: 1).
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

train_movie_id, train_title, train_released_year, train_runtime, train_genre, train_imdb_rating, train_director = next(iter(train_dataloader))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")
print("")

model = NeuralNetwork().to(device)

# Perform a transform on the data for it to be usable for the model
X = train_movie_id.unsqueeze(0)
X = X.to(device)
X = X.type(torch.float32)

print(f"Shape: {X.shape}")
print(f"Datatype: {X.dtype}")
print(f"Device: {X.device}")
print("")

logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
pred_tensor = X[0][y_pred]
print(f"Predicted class: {y_pred}")
print("")

movie_id = int(pred_tensor[0])
_, title, released_year, runtime, genre, imdb_rating, director = dataset.__getitem__(movie_id)

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
