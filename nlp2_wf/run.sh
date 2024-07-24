#!/bin/bash

python main.py --model LSTM
python main.py --model BiLSTM
python main.py --model dot
# python main.py --model bilinear
# python main.py --model multi_layer_perceptron
python main.py --model transformer
python main.py --model GRU