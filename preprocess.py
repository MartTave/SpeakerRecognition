import librosa.display
import numpy
import matplotlib.pyplot as plt
import soundfile as sf
# Here are the params to tweak
db_filter = 15
filePath = "./audio_sample/Yacine/Yacine.wav"

plt.figure()
x, sr = librosa.load(filePath)
x = librosa.to_mono(x)
ts = librosa.effects.split(x, top_db=db_filter)
splitted = []
for start_i, end_i in ts:
    #librosa.display.waveshow(x[start_i:end_i])
    y = x[start_i:end_i]
    splitted.extend(y)
librosa.display.waveshow(numpy.array(splitted))
# Write out audio as 24bit PCM WAV
sf.write('./out/processed.wav', splitted, sr, subtype='PCM_24')
plt.show()