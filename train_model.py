"""Training a Movie Model"""

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import L1Loss

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from ml.dataset import MovieDataset
from ml.model import MovieRecommendation

from config import CSV_DATASET, CSV_VALIDATION_DATASET

TRAINED_MODEL_DIR = "trained_models/"
LOG_DATA_DIR = "runs/"

# batch_size - how many samples per batch to load
BATCH_SIZE = 4
EPOCHS = 160
LEARNING_RATE = 0.0000005


def train_one_epoch(
    epoch_index: int,
    tb_writer: SummaryWriter,
    training_loader: DataLoader,
    optimizer: SGD,
    device: str,
    model: MovieRecommendation,
    loss_fn: L1Loss,
):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        movie_ids, labels = data

        # Zero the gradients for every batch!
        optimizer.zero_grad()

        # Perform a transform on the data for it to be usable for the model
        movie_ids = movie_ids.to(device)
        movie_ids = movie_ids.type(torch.float32)

        labels = labels.to(device)
        labels = labels.type(torch.float32)

        # Make predictions for this batch
        outputs = model(movie_ids)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

        return last_loss


def main():
    training_set = MovieDataset(CSV_DATASET)
    validation_set = MovieDataset(CSV_VALIDATION_DATASET)

    training_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False
    )

    # Report the sizes of the datasets
    print(f"Training set has {len(training_set)} instances")
    print(f"Validation set has {len(validation_set)} instances")
    print("")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using {device} device")
    print("")

    # Model
    model = MovieRecommendation().to(device)

    # Loss Function
    loss_fn = torch.nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOG_DATA_DIR + f"movie_trainer_{timestamp}")
    epoch_number = 0
    best_vloss = 1_000_000.0

    for _ in range(EPOCHS):
        print(f"EPOCH {epoch_number + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch_number,
            writer,
            training_loader,
            optimizer,
            device,
            model,
            loss_fn,
        )

        running_vloss = 0.0

        # Set model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for _, vdata in enumerate(validation_loader):
                vmovie_ids, vlabels = vdata

                vmovie_ids = vmovie_ids.to(device)
                vmovie_ids = vmovie_ids.type(torch.float32)

                vlabels = vlabels.to(device)
                vlabels = vlabels.type(torch.float32)

                voutputs = model(vmovie_ids)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (len(validation_loader))
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        # Log the running loss average per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # Save the model's state
            model_path = (
                TRAINED_MODEL_DIR + f"model_{timestamp}_{epoch_number}.pt"
            )
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    main()
