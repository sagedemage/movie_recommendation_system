import torch
from torch.utils.data import DataLoader
from datetime import datetime
import math

from torch.utils.tensorboard import SummaryWriter

from ml_model.dataset import CustomDataset
from ml_model.neural_network import NeuralNetwork

TRAINED_MODEL_DIR = "trained_models/"

csv_dataset = 'data/imdb_top_1000.csv'
csv_validation_dataset = 'validation_data/favorite_movies_imdb.csv'
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

training_set = CustomDataset(csv_dataset)
validation_set = CustomDataset(csv_validation_dataset)

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
best_vloss =1.414661889*math.pow(10, 35)

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
    if avg_vloss <= best_vloss:
        best_vloss = avg_vloss
        model_path = TRAINED_MODEL_DIR + 'model_{}_{}.pt'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
