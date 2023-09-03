import librosa.display
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from classes.sample import Sample
from typing import List
from classes.timestamp import Timestamp



def getTimeFromMfcc(i):
    # TODO: Transform from mfcc index to time
    return 0


predictSample = []

targets = [
    "Paul Mirabel",
    "Yann Marguet"
    "Julien Machin Truc",
    "Yacine Truc Chose"
]


PM = Sample("./out/mirabel_processed.wav")
YM = Sample("./out/marguet_processed.wav")
JM = Sample("./out/julien_processed.wav")
Y = Sample("./out/yacine_processed.wav")


merged, target = Sample.concat()

x_train, x_test, y_train, y_test = train_test_split(merged, target, test_size=0.3, random_state=56)
clf = svm.SVC()
clf.fit(x_train, y_train)



res = clf.predict(x_test)

#labels = clf.predict(predictSample)
# output = ""
# currentActiveSpeaker = -1
# currentActiveSpeakerTime = -1

# def prettyPrint(label, index):
#     startTime = getTimeFromMfcc(currentActiveSpeakerTime)
#     endTime = getTimeFromMfcc(index)
#     output += str(label) + ":" + Timestamp.toString(startTime) + " to " + Timestamp.toString(endTime) + "\n"



# def stopPreviousSpeaker(index, label):
#     if label != -1 and currentActiveSpeaker != -1:
#         prettyPrint(label, index)
#     lastLabel = -1
#     currentActiveSpeaker = -1
#     currentActiveSpeakerTime = -1

# def startNewSpeaker(index, label):
#     currentActiveSpeaker = label
#     currentActiveSpeakerTime = index

# # TODO: Need to implement a SVM to detect when there's someone speaking or not
# # So that we don't treat the silences
# silences = []
# for i in range(0, len(labels)):
#     if silences[i] == 1:
#         # This mean silence
#         stopPreviousSpeaker(i, currentActiveSpeaker)
#     else :
#         if labels[i] != currentActiveSpeaker:
#             # This mean we have a change in active speaker
#             stopPreviousSpeaker(i, currentActiveSpeaker)
#             startNewSpeaker(i, labels[i])
        



res = res.tolist()
print("Accuracy was : ", metrics.accuracy_score(y_test, res))