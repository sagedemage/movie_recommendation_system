# movie_recommendation_system
Movie recommendation system written in Python with PyTorch.

## Source of the IMDB Movies Dataset
[IMDB Movies Dataset - kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

## 1. Create Validation Data

1.1 Change the index of the row to any movie you want in the `generate_validation_data.py` script.
```
...
# Pick a movie
row = df_data.iloc[10]
genre = row[4]
pick_genre_list = genre.split(",")
...
```

1.1.1 For example, change the index of the row to 5:
```
...
# Pick a movie
row = df_data.iloc[5]
...
```

1.2 Create validation data
```
python3 generate_validation_data.py
```

## 2. Train a Model
2.1 Make sure to create the `trained_models` directory before training a model
```
mkdir trained_models
```

2.2 Train a model
```
python3 train_model.py
```

## 3. Run Program
3.1 Run the program with the file path of the model
```
python3 main.py trained_models/model_20241012_191803_0
```

## Resource
- [PyTorch Website](https://pytorch.org)
- Introduction to PyTorch
  - [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
  - [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
  - [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
  - [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- Introduction to PyTorch on YouTube
  - [Introduction to PyTorch - YouTube Series](https://pytorch.org/tutorials/beginner/introyt.html)
  - [Introduction to PyTorch](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)
  - [Introduction to PyTorch Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
  - [The Fundamentals of Autograd](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
  - [Building Models with PyTorch](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
  - [PyTorch TensorBoard Support](https://pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html)
  - [Training with PyTorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
  - [Model Understanding with Captum](https://pytorch.org/tutorials/beginner/introyt/captumyt.html)
