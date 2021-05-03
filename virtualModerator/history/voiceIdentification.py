import librosa
import librosa.display as ld
from librosa import feature
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import *

import seaborn as sns

from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture as GM
from sklearn import svm
import joblib

from spectralcluster import SpectralClusterer
import math

# sampling rate of stored audio files
rate = 22050


# scales and reduces dimensionality of feature vectors
def normalizeFeatures(data, name, predict, visualize=True):
    if predict:
        transformer = joblib.load(f'{name} MaxAbsScaler.pkl')
        data = transformer.transform(data)
        return data
    # scales data
    transformer = MaxAbsScaler().fit(data)
    joblib.dump(transformer, f'{name} MaxAbsScaler.pkl')
    data = transformer.transform(data)
    # visualizes scaled feature spread
    if visualize:
        for i in range(data.shape[1]):
            sns.kdeplot(data[:, i])
        plt.show()
    return data


def preprocess(rawData):
    trimData, _ = librosa.effects.trim(rawData)
    return trimData


def extract_feature(rawSound, mfcc, chroma, mel):
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

# returns feature vector for a sample
def getFeatureVector(rawSound):
    # L = len(rawSound)//6
    # n_fft = 2 ** nextPow2(L)
    # n_fft also used as window size
    n_fft = 2048
    hop = 512

    # features to consider
    fnList1 = [
        feature.chroma_stft,
        feature.spectral_centroid,
        feature.spectral_bandwidth,
        feature.spectral_rolloff,
        feature.melspectrogram,
    ]

    fnList2 = [
        feature.rms,
        feature.zero_crossing_rate,
        feature.spectral_flatness,
    ]

    # creates power spectogram
    D = np.abs(librosa.stft(rawSound, n_fft=n_fft, hop_length=hop))
    fnList3 = [
        np.max(D),
        np.std(D),
        np.mean(D),
        np.min(D)
    ]

    # median is used so as to mitigate effect of outliers
    featList1 = [np.median(funct(rawSound, rate)) for funct in fnList1]
    featList2 = [np.median(funct(rawSound))//len(rawSound) for funct in fnList2]
    # retrieve mfcc for audio data
    mfccs = librosa.feature.mfcc(rawSound, rate, n_mfcc=20, fmax=7000)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    # combine features into single list
    featList = featList1 + featList2 + fnList3
    for coeff in mfccs_scaled:
        featList.append(coeff)
    return featList


# receives array of raw sounds for particular class of sound as input
# returns features of that class
def featurizeInput(typeRawSounds):
    out = []
    for sample in typeRawSounds:
        sample = preprocess(sample)
        print(sample, len(sample))
        fv = extract_feature(sample, False, False, True)
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


def loadSampleFeatures(samples, name, isPredict):
    samples = [librosa.load(sample)[0] for sample in samples]
    splitSamples = []
    for sample in samples:
        binned = windowData(sample)
        print(len(binned))
        splitSamples.extend(binned)
    # extract features from samples
    features = featurizeInput(splitSamples)
    features = normalizeFeatures(features, name=name, predict=isPredict, visualize=False)
    return features


def saveClfSingle(name, sampleFeats):
    clf1 = svm.OneClassSVM(nu=.1, kernel="rbf", gamma=.1)
    clf1.fit(sampleFeats)
    clf2 = GM(n_components=2, covariance_type='diag', n_init=3)
    clf2.fit(sampleFeats)
    joblib.dump(clf1, f'{name} clf.pkl')
    joblib.dump(clf2, f'{name} GMM clf.pkl')


def trainVoiceIdentification(name, trainSamples):
    trainFeatures = loadSampleFeatures(trainSamples, name, isPredict=False)
    saveClfSingle(name, trainFeatures)


# splits single audio sample into multiple samples of window size
def windowData(rawData,  windowSize=rate):
    return [rawData[x:x + windowSize] for x in range(0, len(rawData), windowSize)]


def clusterData(testSamples):
    name = 'Jett'
    testSamples = loadSampleFeatures(testSamples, name, isPredict=True)
    clusterer = SpectralClusterer(
        min_clusters=1,
        max_clusters=2,
        p_percentile=0.95,
        gaussian_blur_sigma=1)
    labels = clusterer.predict(testSamples)
    return labels


def testVoiceIdentification(name, testSamples):
    fileName = f'{name} clf.pkl'
    fileNameGM = f'{name} GMM clf.pkl'
    predictions = []
    print(f'Loading {fileName}...')
    clf = joblib.load(fileName)
    print(f'{fileName} loaded.')
    print('Loading test features...')
    testSamples = loadSampleFeatures(testSamples, name, isPredict=True)
    print(testSamples)
    print('Test features loaded.')
    clusterer = SpectralClusterer(
        min_clusters=1,
        max_clusters=100,
        p_percentile=0.95,
        gaussian_blur_sigma=1)
    featList = []
    for fv in testSamples:
        featList.append(fv)
        print(f'-->{fv}')
        pred = clf.predict([fv])
        print(type(pred))
        conf = clf.decision_function([fv])[0]
        # -1 = predicted speaker is NOT trained speaker
        if pred[0] == -1:
            predicted = False
        # 1 = predicted speaker is trained speaker
        if pred[0] == 1:
            predicted = True
        predictions.append((conf, predicted))
    featList = np.array(featList)
    print(clusterer.predict(featList))
    return predictions


print(clusterData(['virtualModerator\\test audio\\identification\\jett 1.wav']))


# trainSamples = ['virtualModerator\\test audio\\identification\\jett 1.wav', 'virtualModerator\\test audio\\identification\\jett 2.wav']
# trainVoiceIdentification('Jett', trainSamples)

# testSamples = ['virtualModerator\\test audio\\identification\\jett test 2.wav', 'virtualModerator\\test audio\\identification\\jett 2.wav', 'virtualModerator\\test audio\\identification\\jett 1.wav', 'virtualModerator\\test audio\\identification\\drake.wav', 'virtualModerator\\test audio\\identification\\jett test 1.wav']
# print(testVoiceIdentification('Jett', testSamples))

# test = librosa.load('virtualModerator\\test audio\\identification\\jett 1.wav')
# print(test[0])
# print(preprocess(test[0]))


# accuracies = classify(useStored=True, store=True)
#
# for clf in accuracies:
#     print(f"{clf} Mean Accuracy: {accuracies[clf]}")

