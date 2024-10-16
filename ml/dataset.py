import pandas as pd
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, file_name: str, transform=None, target_transform=None):
        self.df_data = pd.read_csv(file_name)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx: int):
        row = self.df_data.iloc[idx]

        movie_id = row["Movie_ID"]
        title = row["Series_Title"]
        released_year = row["Released_Year"]
        runtime = row["Runtime"]
        genre = row["Genre"]
        imdb_rating = row["IMDB_Rating"]
        director = row["Director"]

        if self.target_transform:
            movie_id = self.target_transform(movie_id)
            title = self.target_transform(title)
            released_year = self.target_transform(released_year)
            runtime = self.target_transform(runtime)
            genre = self.target_transform(genre)
            imdb_rating = self.target_transform(imdb_rating)
            director = self.target_transform(director)

        return movie_id, title, released_year, runtime, genre, imdb_rating, director

    def get_item_by_movie_id(self, movie_id: int):
        rows = self.df_data.loc[self.df_data["Movie_ID"] == movie_id]
        row = rows.iloc[0]

        title = row["Series_Title"]
        released_year = row["Released_Year"]
        runtime = row["Runtime"]
        genre = row["Genre"]
        imdb_rating = row["IMDB_Rating"]
        director = row["Director"]

        if self.target_transform:
            #movie_id = self.target_transform(movie_id)
            title = self.target_transform(title)
            released_year = self.target_transform(released_year)
            runtime = self.target_transform(runtime)
            genre = self.target_transform(genre)
            imdb_rating = self.target_transform(imdb_rating)
            director = self.target_transform(director)

        return title, released_year, runtime, genre, imdb_rating, director