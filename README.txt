HW3P2
Yijin Huang, yijinh@andrew.cmu.edu

# Description
## Data loading
rnn_utils.pad_sequence is applied to align data in one batch. Z-score normalization is applied in data preprocessing.

## Model
A normal 5-layer bidirectional LSTM net with dropout of 0.2 probability is adopted as backbone. 2-layer fully connected layer with ReLU activate function is used to map feature to prediction.

## Configurations
Nesterov adam optimizer with momentum of 0.9 is adopted in this work. Initial learning rate is 0.002 and cosine decay schedule is applied during training phase. 30 epochs with batch size 64 are used to train the model.

# How to use

## Setting
You can update your training configurations and hyperparameters including context in the 'config.py'.

## Run
Run the following code to train the LSTM.
$ python main.py

# Note
Part of the code is written based on *my own* public github repository: https://github.com/YijinHuang/pytorch-classification

