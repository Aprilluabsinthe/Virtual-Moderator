import librosa
import librosa.display as ld
from librosa import feature
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import *

import random

import seaborn as sns

from sklearn import svm
from sklearn.mixture import GaussianMixture as GM
import joblib

from spectralcluster import SpectralClusterer
import math

# sampling rate of stored audio files
rate = 22050


# scales and reduces dimensionality of feature vectors
def normalizeFeatures(data, visualize=True):
    # scales data
    transformer = MaxAbsScaler().fit(data)
    data = transformer.transform(data)
    # visualizes scaled feature spread
    if visualize:
        for i in range(data.shape[1]):
            sns.kdeplot(data[:, i])
        plt.show()
    return data


def preprocess(rawData):
    # given sampling rate, will split audio
    # by detecting audio lower than 40db for a period of 1 sec
    trimData, _ = librosa.effects.trim(rawData, frame_length=rate, top_db=40)
    return trimData


def extractFeature(rawSound, mfcc=False, chroma=False, mel=True):
        X = rawSound
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=rate).T,axis=0)
            result=np.hstack((result, mel))
        return result


# receives array of raw sounds for particular class of sound as input
# returns features of that class
def featurizeInput(typeRawSounds):
    out = []
    for sample in typeRawSounds:
        sample = preprocess(sample)
        print(sample, len(sample))
        fv = extractFeature(sample, False, False, True)
        # fv = getFeatureVector(sample)
        out.append(fv)
    out = np.array(out)
    return out


def plotWaves(raw_sounds):
    i = 1
    for f in raw_sounds:
        plt.subplot(len(raw_sounds),1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        # plt.title(f"{i}")
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()


# create wave, HZ, and power spec charts for a single file
def visualizeIndividualFile(rawData):
    ld.waveplot(rawData, sr=rate)

    plt.show()

    trimData, _ = librosa.effects.trim(rawData)
    n_fft = 2048

    hop_length = 512
    D = np.abs(librosa.stft(trimData, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    plt.plot(D)
    plt.show()

    librosa.display.specshow(DB, sr=rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def loadSampleFeatures(sampleIn):
    sample = librosa.load(sampleIn)[0]
    splitSamples = []
    binned = windowData(sample)
    print(len(binned))
    splitSamples.extend(binned)
    # extract features from samples
    features = featurizeInput(splitSamples)
    features = normalizeFeatures(features, visualize=False)
    return features


# splits single audio sample into multiple samples of window size
def windowData(rawData,  windowSize=rate):
    return [rawData[x:x + windowSize] for x in range(0, len(rawData), windowSize)]


def clusterRealTime(features, numSpeakers=2):
    clusterer = SpectralClusterer(
        min_clusters=numSpeakers,
        max_clusters=numSpeakers,
        p_percentile=0.95,
        gaussian_blur_sigma=1)
    labels = clusterer.predict(features)
    diarized = getTimesFromLabels(labels)
    print(diarized)
    return labels, diarized


def clusterData(testSampleOg, numSpeakers=2):
    testSample = loadSampleFeatures(testSampleOg)
    clusterer = SpectralClusterer(
        min_clusters=numSpeakers,
        max_clusters=numSpeakers,
        p_percentile=0.95,
        gaussian_blur_sigma=1)
    labels = clusterer.predict(testSample)
    diarized = getTimesFromLabels(labels)
    print(diarized)
    visualizeDiarized(testSampleOg, diarized)
    return labels, diarized


def getTimesFromLabels(labels):
    timeDict = dict()
    start = 0
    for i in range(1,len(labels)):
        previous = labels[i-1]
        current = labels[i]
        # cases for sequence at end of labels
        if i == len(labels) - 1:
            if current != previous:
                timeDict[current] = timeDict.get(current, [])
                timeDict[current].append((i, i))
            else:
                timeDict[current] = timeDict.get(current, [])
                timeDict[current].append((start, i))
        if current != previous:
            timeDict[previous] = timeDict.get(previous, [])
            timeDict[previous].append((start, i-1))
            start = i
    return timeDict


def randomColor():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])


def visualizeDiarized(sample, diarizedDict):
    ld.waveplot(librosa.load(sample)[0], sr=rate)
    fileName = os.path.basename(sample)
    plt.title(f'{fileName} Diarized')
    patches = []
    for i, label in enumerate(diarizedDict):
        times = diarizedDict[label]
        color = randomColor()
        patches.append(mpatches.Patch(color=color, label=f'Spaker {i}'))
        for start,end in times:
            plt.axvspan(start, end+1, color=color, alpha=0.5)
        plt.legend(handles=patches)
    plt.show()


# result = clusterData('virtualModerator\\test audio\\identification\\zoom.wav', 3)
#
# print(result)
