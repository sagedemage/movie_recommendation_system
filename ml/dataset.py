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

        title = row["Series_Title"]
        released_year = row["Released_Year"]
        runtime = row["Runtime"]
        genre = row["Genre"]
        imdb_rating = row["IMDB_Rating"]
        director = row["Director"]

        if self.target_transform:
            title = self.target_transform(title)
            released_year = self.target_transform(released_year)
            genre = self.target_transform(genre)
            imdb_rating = self.target_transform(imdb_rating)
            director = self.target_transform(director)

        return idx, title, released_year, runtime, genre, imdb_rating, director
