"""Measure the accuracy of the trained model"""

import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Missing the model file path!")
        exit()
    args = sys.argv
    # Use trained model
    model_path = args[1]

    file = open("validation_data/picked_movie.txt", "r", encoding="utf-8")
    genre_list = []
    for line in file:
        if line[0:6] == "Genre:":
            line = line.replace("Genre:", "")
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            genre_list = line.split(",")
    file.close()
    print(genre_list)

    correct = 0

    for _ in range(100):
        result = os.popen("python3 main.py " + model_path)

        r_genre_list = []

        for line in result:
            if line[0:6] == "Genre:":
                line = line.replace("Genre:", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                r_genre_list = line.split(",")
                break

        for j in range(len(r_genre_list)):
            if r_genre_list[j] in genre_list:
                correct += 1
                break

    acc_per = correct
    print(f"Accuracy of {acc_per}%")


if __name__ == "__main__":
    main()
