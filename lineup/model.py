#!/usr/bin/env python

import os
import h5py
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import tensorflow as tf

import keras.backend as K
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model, model_from_json
from keras.utils import plot_model

def build_model(lstm_shape):
    def crop(dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)

    print('Compiling model...')
    # LSTM input [samples, timesteps, features] --> (timesteps, features)
    time_input = Input(shape=lstm_shape, dtype='float32', name='lstm_input')
    lstm1 = LSTM(100)(time_input)
    hidden1 = Dense(50, activation='relu')(lstm1)
    hidden2 = Dense(50, activation='relu')(hidden1)
    output = Dense(1, activation='softmax')(hidden2)
    model = Model(inputs=time_input, outputs=output)
    
    return model

def compile_model(model):
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    return model

def load_model():
    print("Loading model from disk")
    with open('model.json', "r") as jfopen:
        loaded_model_json = jfopen.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model_weights.hd5')

    return loaded_model

def save_model(trained_model):
    print('Saving model...')
    model_json = trained_model.to_json()
    with open('model.json', "w+") as jfopen:
        jfopen.write(model_json)
    trained_model.save_weights('model_weights.hd5')

def save_model_summary(model, time_str):
    directory = 'model_history/{}'.format(time_str)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('{}/model_summary.txt'.format(directory),
        'w+') as fopen:
        model.summary(print_fn=lambda x: fopen.write(x + '\n'))
    plot_model(model, to_file='{}/model_struct.png'.format(directory))

def save_history(history, time_str):
    print('Saving model history (acc, loss)')
    with open('model_history/{}/model_history.pkl'.format(time_str), 
        'wb+') as fopen:
        pickle.dump(history.history, fopen, protocol=pickle.HIGHEST_PROTOCOL)

def save_plot_history(history, time_str):
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_history/{}/model_image.png'.format(time_str))

def train_model(train_X, train_y, test_X, test_y, compiled_model, epochs, 
    batch_size):
    history = compiled_model.fit(train_X, train_y, 
        epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), 
        verbose=2, shuffle=False)

    return compiled_model, history

def run_train(epochs, batch_size=500):
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    with h5py.File('../data/model_data.h5','r') as h5f:
        train_X = h5f['train_X'][:]
        test_X = h5f['test_X'][:]
        train_y = h5f['train_y'][:]
        test_y = h5f['test_y'][:]

    lstm_shape = train_X.shape[1:]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.805)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options)) as sess:
        K.set_session(sess)
        model = build_model(lstm_shape)
        save_model_summary(model, time_str)
        compiled_model = compile_model(model)
        trained_model, history = train_model(train_X, train_y, test_X, 
            test_y, compiled_model, epochs=epochs, batch_size=batch_size)
        save_model(trained_model)
        save_history(history, time_str)
        save_plot_history(history, time_str)
    K.clear_session()

def test_model(compiled_model, input_data):
    return compiled_model.predict(input_data)

def run_test():
    input_data = None
    model = load_model()
    compiled_model = compile_model(model)
    test_model(compiled_model, input_data)
    K.clear_session()

if __name__ == '__main__':
    run_train(epochs=10, batch_size=500)