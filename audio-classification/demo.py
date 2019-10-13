
from azure.storage.blob import BlockBlobService, PublicAccess
from playsound import playsound
from keras.models import load_model
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import json
import csv

# gets rid of warning about unused cpu instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def demo(audio_path):
    y_true = list()
    y_pred = list()
    file_name_prob = dict()
    files = list([match for match in os.listdir(audio_path) if "baby" in match])
    for file in files:
        print("PLAYING[{}]".format(file))
        #playsound(os.path.join(audio_path, file))
    print("Extracting features from audio")
    for file in tqdm(files):
        rate, wav = wavfile.read(os.path.join(audio_path, file))
        label = file_2_class[file]
        class_index = classes.index(label)
        y_prob = list()
        for index in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[index:index+config.step]
            x = mfcc(sample, rate, numcep=config.features, nfilt=config.filters, 
                nfft=config.fourier_transforms)
            x = (x-config.min)/(config.max-config.min)
            if config.mode == "cnn":
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == "time":
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(class_index)
        file_name_prob[file] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, file_name_prob

data_frame = pd.read_csv("demo.csv")
classes = list(np.unique(data_frame.label))
file_2_class = dict(zip(data_frame.fname, data_frame.label))
pickle_path = os.path.join("pickles", "cnn.p")

with open(pickle_path, "rb") as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, file_name_prob = demo("clean")
accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = list()
for index, row in data_frame.iterrows():
    y_prob = file_name_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        data_frame.at[index, c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
data_frame["y_pred"] = y_pred
data_frame.to_csv("demo.csv", index=False)

data = {}
with open('demo.csv') as csv_data:
    reader = csv.reader(csv_data)
    rows = [row for row in reader if row]
    headings = rows[0]

    for row in rows[1:]:
        for col_header, data_column in zip(headings, row):
            data.setdefault(col_header, []).append(data_column)

blob_service = BlockBlobService('lullaby','')

blob_service.create_container(
    'lullaby',
    public_access=PublicAccess.Blob
)

blob_service.create_blob_from_bytes(
    'lullaby',
    'lullaby-blob',
    json.dumps(data).encode("utf-8")
)

print(blob_service.make_blob_url('lullaby', 'lullaby-blob'))
