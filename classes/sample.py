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
    def __init__(self, filePath, test_sample = False) -> None:
        self.data, sr = librosa.load(filePath)
        self.data = librosa.feature.mfcc(y=self.data, sr=sr, htk=True, lifter=80, n_mfcc=40)
        self.data = numpy.transpose(self.data)
        if not test_sample:
            self.target = [Sample.currentId] * len(self.data)
            Sample.samples.append(self)
            Sample.currentId += 1