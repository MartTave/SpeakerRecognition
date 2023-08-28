import librosa.display
import numpy as np
from sklearn import svm

x, sr = librosa.load("./out/marguet_processed.wav")
y, sr2 = librosa.load("./out/mirabel_processed.wav")

#len = 10491
mfcc = librosa.feature.mfcc(y = x)
mfcc = np.transpose(mfcc, (1, 0))

trainData = mfcc[:-1000]
testData = mfcc[-1000:]


mfcc2 = librosa.feature.mfcc(y = y)
mfcc2 = np.transpose(mfcc2, (1, 0))

trainData2 = mfcc2[:-1000]
testData2 = mfcc2[-1000:]

# 1 if from sample n°1
target = [1] * len(trainData)
# 2 if from sample n°2
target.extend([0] * len(trainData2))
merged = []
merged.extend(trainData)
merged.extend(trainData2)


clf = svm.SVC()
clf.fit(merged, target)

finalTarget = [1] * 1000
finalTarget.extend([0] * 1000)
finalTestData = []
finalTestData.extend(testData)
finalTestData.extend(testData2)


res = clf.predict(finalTestData)

found = 0
total = 0
for i in range(0, len(res)):
    if res[i] == finalTarget[i]:
        found += 1
    total += 1
print("Found ratio was : ", (found / total * 100), "%")