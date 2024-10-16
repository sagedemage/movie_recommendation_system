# movie_recommendation_system
Movie recommendation system written in Python with PyTorch.

## Source of the IMDB Movies Dataset
[IMDB Movies Dataset - kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

## Usage Instructions
### 1. Create Validation Data
1.1 Create validation data with the index of the row:
```
python3 generate_validation_data.py 21
```

### 2. Train a Model
2.1 Make sure to create the `trained_models` directory before training a model
```
mkdir trained_models
```

2.2 Start TensorBoard
```
tensorboard --logdir=runs
```

2.3 Train a model
```
python3 train_model.py
```

### 3. Run Program
3.1 Run the program with the file path of the model
```
python3 main.py trained_models/model_20241012_195623_0.pt
```

## Accuracy of the Model

A user chooses this as their favorite movie:
- Movie_ID: 22
- Series_Title: Cidade de Deus
- Released_Year: 2002
- Certificate: A
- Runtime: 130 min
- Genre: Crime, Drama
- IMDB_Rating: 8.6
- Overview: In the slums of Rio, two kids' paths diverge a...
- Meta_score: 79.0
- Director: Fernando Meirelles
- Star1: Kátia Lund
- Star2: Alexandre Rodrigues
- No_of_Votes: 699256
- Gross: 7563397.0

### 1. Results
Here are the results after running the movie recommendation program 100 times:
- Accuracy of the recommendations: 78%

**Note**: The `measure_accuracy.py` script runs the program 100 times for me.

The accuracy of the model is 78%. There is certainly more work that has to be
done to get it to a 90% accuracy. An accuracy above 75% is good for a demo
project.

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
