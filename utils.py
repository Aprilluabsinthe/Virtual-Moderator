import numpy as np
import struct
import random
from tkinter import filedialog
import pyttsx3
import csv
import string


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype='int16').tobytes()


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def floatArraytoPCM(toConvert):
    samples = [sample * 32767
               for sample in toConvert]
    return struct.pack("<%dh" % len(samples), *samples)


def getAudioPath():
    fileName = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .ogg"), ("All files", "*.*")))
    return fileName


def randomColors(numSpeakers):
    distinctColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                      '#ffffff', '#000000']
    random.shuffle(distinctColors)
    return distinctColors[0:numSpeakers]


# enables audio notifications
def verbalSuggestions(cues, isMale=False, rate=146):
    if isMale:
        isMale = 0
    else:
        isMale = 1
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', rate)
    engine.setProperty('voice', voices[isMale].id)
    for cue in cues:
        # que cue
        engine.say(cue)
    engine.runAndWait()


class Conll:
    posTagsDict = {'B-INTJ': 0, 'B-LST': 1, 'B-PRT': 2, 'I-UCP': 3, 'B-CONJP': 4, 'B-NP': 5, 'I-ADVP': 6, 'I-PP': 7,
                   'I-INTJ': 8, 'B-SBAR': 9, 'B-PP': 10, 'I-SBAR': 11, 'B-VP': 12, 'B-ADJP': 13, 'I-NP': 14,
                   'B-UCP': 15, 'I-PRT': 16, 'O': 17, 'I-CONJP': 18, 'I-ADJP': 19,
                   'I-VP': 20, 'B-ADVP': 21, 'I-LST': 22}
    testChunks = []
    trainChunks = []


class Chunk:
    def __init__(self, chunk, wsjLabels, posLabels):
        self.fullText = chunk
        self.wsjLabels = wsjLabels
        self.posLabels = posLabels


def readTestConll():
    testChunk = ''
    testWsjLabels = []
    testPosLabels = dict()
    with open('language/language data sets/conll 2000 test.txt', newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=' ')
        for row in dataReader:
            if not row:
                # store compiled chunk...reset holders
                Conll.testChunks.append(Chunk(chunk=testChunk, wsjLabels=testWsjLabels, posLabels=testPosLabels))
                testChunk = ''
                testWsjLabels = []
                testPosLabels = dict()
            else:
                # update current chunk
                word = row[0]
                wsjLabel = row[1]
                posLabel = row[2]
                if testChunk == '':
                    testChunk = word
                else:
                    testChunk += f' {word}'
                testWsjLabels.append(wsjLabel)
                testPosLabels[word] = Conll.posTagsDict[posLabel]


def readTrainConll():
    trainChunk = ''
    trainWsjLabels = []
    trainPosLabels = dict()
    with open('language/language data sets/conll 2000 train.txt', newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=' ')
        for row in dataReader:
            if not row:
                # store compiled chunk...reset holders
                Conll.trainChunks.append(Chunk(chunk=trainChunk, wsjLabels=trainWsjLabels, posLabels=trainPosLabels))
                trainChunk = ''
                trainWsjLabels = []
                trainPosLabels = dict()
            else:
                # update current chunk
                word = row[0]
                wsjLabel = row[1]
                posLabel = row[2]
                if trainChunk == '':
                    trainChunk = word
                else:
                    trainChunk += f' {word}'
                trainWsjLabels.append(wsjLabel)
                trainPosLabels[word] = Conll.posTagsDict[posLabel]


def readConll():
    readTrainConll()
    readTestConll()
    return Conll

