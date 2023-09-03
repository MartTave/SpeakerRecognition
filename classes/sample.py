import librosa
import numpy

class Sample:
    samples = []
    currentId = 0

    @staticmethod
    def concat():
        data = []
        target = []
        for sample in Sample.samples:
            data.extend(sample.data)
            target.extend(sample.target)
        return (data, target)


    data = []
    target = []
    def __init__(self, filePath) -> None:
        self.data, sr = librosa.load(filePath)
        self.data = librosa.feature.mfcc(y=self.data, sr=sr, n_fft=512)
        self.data = numpy.transpose(self.data)
        self.target = [Sample.currentId] * len(self.data)
        Sample.samples.append(self)
        Sample.currentId += 1