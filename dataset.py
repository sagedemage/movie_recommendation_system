import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_name: str, transform=None, target_transform=None):
        self.df_data = pd.read_csv(file_name)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx: int):
        row = self.df_data.iloc[idx]

        column = 0
        title = row[column]

        column = 1
        released_year = row[column]

        column = 3
        runtime = row[column]

        column = 4
        genre = row[column]

        column = 5
        imdb_rating = row[column]

        column = 8
        director = row[column]

        if self.target_transform:
            title = self.target_transform(title)
            released_year = self.target_transform(released_year)
            genre = self.target_transform(genre)
            imdb_rating = self.target_transform(imdb_rating)
            director = self.target_transform(director)

        return idx, title, released_year, runtime, genre, imdb_rating, director
