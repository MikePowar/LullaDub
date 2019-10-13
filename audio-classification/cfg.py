
import os

class config:
    def __init__(self, mode="cnn", filters=26, features=13,
        fourier_transforms=1200, rate=48000):
        self.mode = mode
        self.filters = filters
        self.features = features
        self.fourier_transforms = fourier_transforms
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join("models", mode+".model")
        self.pickle_path = os.path.join("pickles", mode+".p")
