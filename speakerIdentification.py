import numpy as np
import librosa
from sklearn.mixture import GaussianMixture


def extractGMMFeature(rawSound, samplingRate=22050):
    # MFCC configuration:
    N_MFCC = 55
    N_MELS = 15
    WINDOW_LENGTH = int(0.025 * samplingRate)  # To obtain 25 ms window
    HOP_LENGTH = int(0.010 * samplingRate)  # 10 ms shift between consecutive windows

    # Extracting MFCCs:
    mfccs = librosa.feature.mfcc(rawSound, sr=samplingRate, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=WINDOW_LENGTH,
                                 hop_length=HOP_LENGTH).T
    deltaMfcc = librosa.feature.delta(mfccs, order=1)
    deltaMfcc2nd = librosa.feature.delta(mfccs, order=2)
    result = np.hstack((mfccs, deltaMfcc, deltaMfcc2nd))
    return result


def trainGMM(filename, nMixtures=128, samplingRate=22050):
    rawData = librosa.load(filename)[0]
    features = extractGMMFeature(rawData, samplingRate)
    gmm = GaussianMixture(n_components=nMixtures)
    start = 0
    while start + 200 < features.shape[0]:
        gmm.fit(features[start:start + 200, :])
        start += 200
    return gmm


# returns speaker index associated with predicted speaker
def identify(gmms, audioSection, speakerNamesList):
    features = extractGMMFeature(rawSound=audioSection)
    scores = []
    for gmm in gmms:
        scores.append(gmm.score(features))
    # return name of predicted speaker
    return speakerNamesList[scores.index(max(scores))]

