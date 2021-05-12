HW3P2
Yijin Huang, yijinh@andrew.cmu.edu

# Description
## Data loading
rnn_utils.pad_sequence is applied to align data in one batch. Z-score normalization is applied in data preprocessing.

## Method
One BLATM folloed by 3 pLSTM layers each with a time reduction are implemented as the encoder. Lockeddropout of 0.2 probability is adopted in each layer of encoder. Key and value projection function are 2 linear layers with LeakyReLU as activation function. Two LSTM layers are implemented as the dedcoder. The hidden dimension of encoder and decoder is 512 and context dimension for attention module is 512 too. During training, Teacher Forcing with factor that decays from 0.9 to 0.6 are applied to help convergence. Finally, greedy search is adopted as decoding method.

## Configurations
Nesterov adam optimizer with momentum of 0.9 is adopted in this work. Initial learning rate is 0.001 and it decays with factor 0.5 at epoch 50 and 85. 100 epochs with batch size 64 are used to train the model.

# How to use

## Setting
You can update your training configurations and hyperparameters in the 'config.py'.

## Run
Update the data path in 'config.py' and run the following code to train the LSTM.
$ python main.py

# Note
Part of the code is written based on *my own* public github repository: https://github.com/YijinHuang/pytorch-classification

