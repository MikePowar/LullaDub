
from cfg import config
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

# gets rid of warning about unused cpu instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_data():
    if os.path.isfile(config.pickle_path):
        print("Loading existing data for {} model".format(config.mode))
        with open(config.pickle_path, "rb") as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_random_features():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = list()
    y = list()
    mi, ma = float("inf"), -float("inf")
    for _ in tqdm(range(samples)):
        random_class = np.random.choice(class_dist.index, p=probability_dist)
        file = np.random.choice(data_frame[data_frame.label == 
            random_class].index)
        rate, wav = wavfile.read("clean/"+file)
        label = data_frame.at[file, "label"]
        random_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[random_index:random_index+config.step]
        x_sample = mfcc(sample, rate, numcep=config.features, 
            nfilt=config.filters, nfft=config.fourier_transforms)
        mi = min(np.amin(x_sample), mi)
        ma = max(np.amax(x_sample), ma)
        X.append(x_sample)
        y.append(classes.index(label))
    config.min = mi
    config.max = ma
    X, y = np.array(X), np.array(y)
    X = (X-mi)/(ma-mi)
    if config.mode == "cnn":
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == "time":
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=4)
    config.data = (X, y)

    with open(config.pickle_path, "wb") as handle:
        pickle.dump(config, handle)
    return X, y
    

def get_cnn_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", strides=(1,1), 
        padding="same", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu", strides=(1,1),
        padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1,1),
        padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", strides=(1,1),
        padding="same"))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5)) # for flattening behavior
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam",
        metrics=["acc"])
    
    return model
    

data_frame = pd.read_csv('cries.csv')
data_frame.set_index('fname', inplace=True)

for file in data_frame.index:
    rate, signal = wavfile.read('clean/'+file)
    data_frame.at[file, 'length'] = signal.shape[0]/rate

classes = list(np.unique(data_frame.label))
class_dist = data_frame.groupby(['label'])['length'].mean()

# sample size may be different
samples = 2*int(data_frame["length"].sum()/0.1)
probability_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p=probability_dist)

figure, axis = plt.subplots()
axis.set_title('Class Distribution', y=1.08)
axis.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False,
    startangle=90)
axis.axis('equal')
plt.show()

config = config("cnn")

if config.mode == "cnn":
    X, y = build_random_features()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_cnn_model()
    
elif config.mode == "rnn":
    X, y = build_random_features()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_rnn_model()

class_weight = compute_class_weight("balanced", np.unique(y_flat), y_flat)
checkpoint = ModelCheckpoint(config.model_path, monitor="val_acc", verbose=1,
    mode="max", save_best_only=True, save_weights_only=False, period=1)
model.fit(X, y, epochs=10, batch_size=32, shuffle=True, 
    class_weight=class_weight, validation_split=0.1, callbacks=[checkpoint])
model.save(config.model_path)
