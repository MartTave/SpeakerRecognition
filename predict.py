from classes.timestamp import Timestamp
from classes.sample import Sample

import joblib



sampleRate = 22050
mffc_window = 512
SameSpeakerIndexTreshold = 15


# Glob vars
output = ""
currentActiveSpeaker = -1
currentActiveSpeakerTime = -1

# save

# load
clf = joblib.load("./model/model.pkl")
print("Model loaded !")

sample = Sample("./test/Yann_Marguet_montreux.mp3")
print("Sample loaded")

res = clf.predict(sample.data)
print("Predicted without prob")

results = clf.predict_proba(sample.data)
print("Predicted with probs")


def getTimeFromMfcc(i):
    time = i * mffc_window / sampleRate * 1000 # Get time in seconds
    return Timestamp.toString(time=time)

def prettyPrint(label, index):
    global output, currentActiveSpeakerTime
    startTime = getTimeFromMfcc(currentActiveSpeakerTime)
    endTime = getTimeFromMfcc(index)
    output += str(label) + "=" + startTime + "-" + endTime + "\n"



def stopPreviousSpeaker(index, label):
    global currentActiveSpeaker, currentActiveSpeakerTime
    if label != -1 and currentActiveSpeaker != -1:
        prettyPrint(label, index)
    currentActiveSpeaker = -1
    currentActiveSpeakerTime = -1

def startNewSpeaker(index, label):
    global currentActiveSpeaker, currentActiveSpeakerTime
    currentActiveSpeaker = label
    currentActiveSpeakerTime = index

def realResult():
    global currentActiveSpeakerTime, currentActiveSpeaker, res
    sameSpeakerIndex = 0
    for i in range(0, len(res)):
        prob_per_class_dictionary = dict(zip(clf.classes_, results[i]))
        print(prob_per_class_dictionary)
        if res[i] != currentActiveSpeaker:
            if sameSpeakerIndex > SameSpeakerIndexTreshold:
                print("Same speaker was active for long enough", sameSpeakerIndex)
                # This mean we have a change in active speaker
                stopPreviousSpeaker(i, currentActiveSpeaker)
                startNewSpeaker(i, res[i])
            else:
                print("Same speaker wasn't active for long enough", sameSpeakerIndex)
                currentActiveSpeaker = res[i]
                currentActiveSpeakerTime = i
            sameSpeakerIndex = 0
        else:
            sameSpeakerIndex += 1
    f = open("./test/result.txt", "w")
    f.write(output)
    f.close()

realResult()