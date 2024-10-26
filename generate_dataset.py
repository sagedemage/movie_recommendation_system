"""Generate a modified dataset from the original dataset"""

import pandas as pd

from config import CSV_DATASET

CSV_ORIGINAL_DATASET = "original_dataset/imdb_top_1000.csv"


def get_label_number(r_genre_item: str):
    match r_genre_item:
        case "Action":
            return 0
        case "Action Epic":
            return 1
        case "Adult Animation":
            return 2
        case "Adventure":
            return 3
        case "Adventure Epic":
            return 4
        case "Animal Adventure":
            return 5
        case "Animation":
            return 6
        case "Anime":
            return 7
        case "Artificial Intelligence":
            return 8
        case "Biography":
            return 9
        case "Boxing":
            return 10
        case "Buddy Comedy":
            return 11
        case "Caper":
            return 12
        case "Car Action":
            return 13
        case "Classical Musical":
            return 14
        case "Comedy":
            return 15
        case "Coming-of-Age":
            return 16
        case "Computer Animation":
            return 17
        case "Conspiracy Thriller":
            return 18
        case "Cop Drama":
            return 19
        case "Costume Drama":
            return 20
        case "Crime":
            return 21
        case "Cyberpunk":
            return 22
        case "Dark Comedy":
            return 23
        case "Dark Fantasy":
            return 24
        case "Dark Romance":
            return 25
        case "Desert Adventure":
            return 26
        case "Docudrama":
            return 27
        case "Drama":
            return 28
        case "Drug Crime":
            return 29
        case "Dystopian Sci-Fi":
            return 30
        case "Epic":
            return 31
        case "Fairy Tale":
            return 32
        case "Fantasy":
            return 33
        case "Fantasy Epic":
            return 34
        case "Family":
            return 35
        case "Farce":
            return 36
        case "Feel-Good Romance":
            return 37
        case "Film-Noir":
            return 38
        case "Financial Drama":
            return 39
        case "Gangster":
            return 40
        case "Globetrotting Adventure":
            return 41
        case "Hand-Drawn Animation":
            return 42
        case "Hard-boiled Detective":
            return 43
        case "Heist":
            return 44
        case "High-Concept Comedy":
            return 45
        case "History":
            return 46
        case "Historical Epic":
            return 47
        case "Holiday":
            return 48
        case "Horror":
            return 49
        case "Jukebox Musical":
            return 50
        case "Jungle Adventure":
            return 51
        case "Legal Drama":
            return 52
        case "Medical Drama":
            return 53
        case "Monster Horror":
            return 54
        case "Music":
            return 55
        case "Musical":
            return 56
        case "Mystery":
            return 57
        case "One-Person Army Action":
            return 58
        case "Quest":
            return 59
        case "Quirky Comedy":
            return 60
        case "Parody":
            return 61
        case "Period Drama":
            return 62
        case "Police Procedural":
            return 63
        case "Political Drama":
            return 64
        case "Political Thriller":
            return 65
        case "Prison Drama":
            return 66
        case "Psychological Drama":
            return 67
        case "Psychological Thriller":
            return 68
        case "Romance":
            return 69
        case "Romantic Comedy":
            return 70
        case "Road Trip":
            return 71
        case "Samurai":
            return 72
        case "Satire":
            return 73
        case "Sea Adventure":
            return 74
        case "Sci-Fi":
            return 75
        case "Sci-Fi Epic":
            return 76
        case "Screwball Comedy":
            return 77
        case "Serial Killer":
            return 78
        case "Showbiz Drama":
            return 79
        case "Slapstick":
            return 80
        case "Space Sci-Fi":
            return 81
        case "Spaghetti Western":
            return 82
        case "Sport":
            return 83
        case "Spy":
            return 84
        case "Steampunk":
            return 85
        case "Superhero":
            return 86
        case "Supernatural Fantasy":
            return 87
        case "Suspense Mystery":
            return 88
        case "Sword & Sandal":
            return 89
        case "Sword & Sorcery":
            return 90
        case "Thriller":
            return 91
        case "Time Travel":
            return 92
        case "Tragedy":
            return 93
        case "True Crime":
            return 94
        case "War":
            return 95
        case "War Epic":
            return 96
        case "Western":
            return 97
        case "Western Epic":
            return 98
        case "Whodunnit":
            return 99
        case _:
            print("Genre is not labeled!")
    return -1


def main():
    # 1. Read the original dataset
    df_data = pd.read_csv(CSV_ORIGINAL_DATASET)

    # 2. Set the columns for the written data
    write_data = {}
    columns = df_data.columns

    ignore_columns = ["Poster_Link", "Star3", "Star4"]
    for i in range(len(columns)):
        column = columns[i]
        if column in ignore_columns:
            continue
        write_data[column] = []

    write_data["Movie_ID"] = []
    write_data["Label"] = []

    # 3. Set the movie id and label for each movie entry
    for i, row in df_data.iterrows():
        r_genre = row["Genre"]
        r_genre = r_genre.replace(" ", "")
        r_genre_list = r_genre.split(",")
        r_genre_item = r_genre_list[0]

        # Specify the label of the movie entry
        label_num = get_label_number(r_genre_item)
        write_data["Label"].append(label_num)

        # Specify the movie id of the movie entry
        write_data["Movie_ID"].append(i)

        # Keep the same values of the rest of the columns
        for j in range(len(columns)):
            column = columns[j]
            if column in ignore_columns:
                continue
            write_data[column].append(row[column])

    # 4. Write the dataset to a CSV file.
    df_write_data = pd.DataFrame(write_data)
    df_write_data.to_csv(CSV_DATASET, index=False)

    print(f"Written the csv file to {CSV_DATASET}")


if __name__ == "__main__":
    main()
