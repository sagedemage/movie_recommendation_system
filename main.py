import torch
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataset import CustomDataset
from neural_network import NeuralNetwork

TRAINED_MODEL_DIR = "trained_models/"

csv_dataset_file = 'data/imdb_top_1000.csv'
MODEL_PATH = TRAINED_MODEL_DIR + "model_20241012_141003_0"
BATCH_SIZE = 64

def train_one_epoch(epoch_index, tb_writer: torch.utils.tensorboard.writer.SummaryWriter):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        movie_id, title, released_year, runtime, genre, imdb_rating, director = data

        # Zero the gradients for every batch!
        optimizer.zero_grad()

        # Perform a transform on the data for it to be usable for the model
        movie_id = movie_id.to(device)
        movie_id = movie_id.type(torch.float32)

        # Make predictions for this batch
        outputs = model(movie_id)

        # Compute the loss and its gradients
        # outputs with a class dimension as [batch_size, nb_classes, *additional_dims]
        loss = loss_fn(outputs, movie_id)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

        return last_loss

training_set = CustomDataset(csv_dataset_file)
validation_set = CustomDataset(csv_dataset_file)

print("Length: " + str(training_set.__len__()))
print("")

movie_id, title, released_year, runtime, genre, imdb_rating, director = training_set.__getitem__(0)
print(f"Movie ID: {movie_id}")
print(f"Title: {title}")
print(f"Released Year: {released_year}")
print(f"Runtime: {runtime}")
print(f"Genre: {genre}")
print(f"IMDB Rating: {imdb_rating}")
print(f"Director: {director}")
print("")

# batch_size (int, optional) – how many samples per batch to load (default: 1).
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

train_movie_id, train_title, train_released_year, train_runtime, train_genre, train_imdb_rating, train_director = next(iter(training_loader))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")
print("")

# Model
model = NeuralNetwork().to(device)

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/movie_trainer_{}'.format(timestamp))
epoch_number = 0
EPOCHS = 5
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    # Set model to evaluation mode
    model.eval()

    # Disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vmovie_id, vtitle, vreleased_year, vruntime, vgenre, vimdb_rating, vdirector = vdata

            vmovie_id = vmovie_id.to(device)
            vmovie_id = vmovie_id.type(torch.float32)

            if vmovie_id.shape[0] != BATCH_SIZE:
                continue

            voutputs = model(vmovie_id)
            vloss = loss_fn(voutputs, vmovie_id)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss average per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch_number + 1)
    writer.flush()

    # Track the best performance, and save the model's state
    #if avg_vloss <= best_vloss:
    best_vloss = avg_vloss
    model_path = TRAINED_MODEL_DIR + 'model_{}_{}'.format(timestamp, epoch_number)
    torch.save(model.state_dict(), model_path)

    epoch_number += 1

# Load a saved version of the model
saved_model = NeuralNetwork().to(device)
saved_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

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
