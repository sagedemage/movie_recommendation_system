import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import L1Loss

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from ml.dataset import CustomDataset
from ml.model import NeuralNetwork

TRAINED_MODEL_DIR = "trained_models/"

csv_dataset = 'data/imdb_top_1000.csv'
csv_validation_dataset = 'validation_data/favorite_movies_imdb.csv'
BATCH_SIZE = 4
EPOCHS = 75

def train_one_epoch(epoch_index, tb_writer: SummaryWriter, training_loader: DataLoader, optimizer: SGD, device: str, model: NeuralNetwork, loss_fn: L1Loss):
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
        # weight, size_average
        loss = loss_fn(outputs, movie_id)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        return last_loss

def main():
    training_set = CustomDataset(csv_dataset)
    validation_set = CustomDataset(csv_validation_dataset)

    print("Length of training set: " + str(training_set.__len__()))
    print("Length of validation set: " + str(validation_set.__len__()))
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
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

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
    loss_fn = torch.nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

    # Training Loop
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/movie_trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 72.19

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_loader, optimizer, device, model, loss_fn)

        running_vloss = 0.0

        # Set model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vmovie_id, vtitle, vreleased_year, vruntime, vgenre, vimdb_rating, vdirector = vdata

                vmovie_id = vmovie_id.to(device)
                vmovie_id = vmovie_id.type(torch.float32)

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

if __name__ == "__main__":
    main()
