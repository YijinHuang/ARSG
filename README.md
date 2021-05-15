## Attention-Based Models for Speech Recognition

### Description
This is the unofficial implementation of **Attention-Based Models for Speech Recognition**. (18660 Optimization project)

### How to use

#### Data

We evaluate our implementation on the public dataset TIMIT. Kaldi is adopted to process audio data to filter bank feature.

#### Setting

You can update your training configurations and hyperparameters in the `config.py`.

#### Run
Update the data path in `config.py` and run the following code to train the ARSG.

```
$ python main.py
```