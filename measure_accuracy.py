"""Measure the accuracy of the trained model"""

import os
import sys
import asyncio


async def measure_accuracy(process_num: int, model_path: str, genre_list: list):
    correct = 0

    for _ in range(10):
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

    accuracy = correct / 10 * 100
    print(f"Process {process_num} done")
    return accuracy


async def main():
    if len(sys.argv) < 2:
        print("Missing the model file path!")
        exit()
    args = sys.argv
    # Use trained model
    model_path = args[1]

    file = open("validation_dataset/picked_movie.txt", "r", encoding="utf-8")
    genre_list = []
    for line in file:
        if line[0:6] == "Genre:":
            line = line.replace("Genre:", "")
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            genre_list = line.split(",")
    file.close()
    print(genre_list)

    accuracy_values = await asyncio.gather(
        measure_accuracy(1, model_path, genre_list),
        measure_accuracy(2, model_path, genre_list),
        measure_accuracy(3, model_path, genre_list),
        measure_accuracy(4, model_path, genre_list),
        measure_accuracy(5, model_path, genre_list),
        measure_accuracy(6, model_path, genre_list),
        measure_accuracy(7, model_path, genre_list),
        measure_accuracy(8, model_path, genre_list),
        measure_accuracy(9, model_path, genre_list),
        measure_accuracy(10, model_path, genre_list),
    )

    accuracy = int(sum(accuracy_values) / 10)
    print(f"Accuracy of {accuracy}%")


if __name__ == "__main__":
    asyncio.run(main())
