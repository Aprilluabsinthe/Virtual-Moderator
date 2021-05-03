import pyaudio
import numpy as np
import time
import matplotlib.pyplot as plt
import speakerIdentification
import matplotlib.patches as mpatches

from six.moves import queue
from google.cloud import speech

import re
import sys
import utils
import threading
from datetime import timedelta

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = CHUNK = 22050


# plt.ion()  # enable interactivity
# fig = plt.figure()  # make a figure


class Conversation:
    """ update to contain a dictionary with time interval as key and speakerID/ words said as values """
    lastSpeaker = None
    # pool of words from current speaker
    speakerPool = []
    # dictionary with start time as key and speakerID/ words said as values
    orderedConversation = {}
    audioSuggestions = False


class Speaker:
    speakerDict = dict()

    def __init__(self, speakerId, color):
        self.speakerID = speakerId
        self.color = color
        self.speechInstances = []
        self.wpm = None
        Speaker.speakerDict[speakerId] = self

    def updateWPM(self):
        times = []

        for instance in self.speechInstances:
            timeDiff = instance.end-instance.start
            times.append(timeDiff.total_seconds())
        if len(times) != 0:
            wpm = (sum(times)/len(times))*60
            self.wpm = wpm


class SpeechInstance:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate):
        self.rate = self.chunk = rate
        # Create a thread-safe buffer of audio data
        self.buff = queue.Queue()
        self.closed = True
        self.storedRaw = []
        self.averageVolume = (0, 0)
        self.gmms = None

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paFloat32,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self.buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self.buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self.buff.get()
            dataFloat = np.frombuffer(chunk, np.float32)

            # update stored volume
            volume = np.median(dataFloat)
            storedVolume, nSamples = self.averageVolume
            self.averageVolume = (storedVolume + volume, nSamples + 1)
            self.storedRaw.append(dataFloat)

            if chunk is None:
                return
            data = [utils.float2pcm(dataFloat)]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    dataFloat = np.frombuffer(chunk, np.float32)
                    # update stored volume
                    volume = np.max(dataFloat)
                    storedVolume, nSamples = self.averageVolume
                    self.averageVolume = (storedVolume + volume, nSamples + 1)
                    self.storedRaw.append(dataFloat)
                    self.storedRaw.append(dataFloat)

                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def transcribe(responses, audioStream):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = ''
        words = []
        for word_info in result.alternatives[0].words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            words.append((word, start_time, end_time))
            transcript += '{} ({}-{}) '.format(word, start_time.total_seconds(), end_time.total_seconds())

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if result.is_final:
            print(f'--->{transcript + overwrite_chars}')
            for word, start, end in words:
                binWord(start, end, word, audioStream)
            updateSpeakerStatistics()
            printReport()
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            num_chars_printed = 0


def updateSpeakerStatistics():
    for speakerID in Speaker.speakerDict:
        speaker = Speaker.speakerDict[speakerID]
        speaker.updateWPM()


# determine if speaker needs to talk louder
def isQuite(sample, audioStream, threshold=.7):
    storedVolume, nSamples = audioStream.averageVolume
    avgMaxVolume = storedVolume/nSamples
    print(f'Avg. Max: {avgMaxVolume*threshold}...Your Volume: {np.median(sample)}')
    # return true if current utterance is lower than average volume
    return np.median(sample) < avgMaxVolume*threshold


def binWord(start, end, word, audioStream):
    sample = audioStream.storedRaw[start.seconds]
    speakerID = speakerIdentification.identify(gmms=audioStream.gmms,
                                               audioSection=sample,
                                               speakerNamesList=list(Speaker.speakerDict))
    if isQuite(sample=sample, audioStream=audioStream):
        if Conversation.audioSuggestions:
            utils.verbalSuggestions([f'You may want to speak up {speakerID}'])
        else:
            print(f'You may want to speak up {speakerID}')
    speaker = Speaker.speakerDict[speakerID]
    currentSpeechInstance = SpeechInstance(start, end, word)
    speaker.speechInstances.append(currentSpeechInstance)
    updateConversation(speakerID=speakerID, currentSpeechInstance=currentSpeechInstance)


# update conversation with new word/ speaker identification
def updateConversation(speakerID, currentSpeechInstance, allowedInterval=5):
    """ separates conversation based on time intervals of who is speaking
    maintains current stream until speakers switch or there is a pause greater than allowed interval"""
    if Conversation.lastSpeaker is None:
        Conversation.lastSpeaker = speakerID
    if len(Conversation.speakerPool) != 0:
        endTime = Conversation.speakerPool[-1].end
        interval = currentSpeechInstance.start-endTime
        interval = interval.total_seconds()
    else:
        interval = 0
        endTime = None
    if speakerID == Conversation.lastSpeaker and interval < allowedInterval:
        Conversation.speakerPool.append(currentSpeechInstance)
    else:
        startTime = Conversation.speakerPool[0].start
        Conversation.orderedConversation[startTime.total_seconds()] = (endTime.total_seconds(),
                                                                       Conversation.lastSpeaker,
                                                                       Conversation.speakerPool)
        Conversation.lastSpeaker = speakerID
        Conversation.speakerPool = [currentSpeechInstance]


def printReport():
    for speakerID in Speaker.speakerDict:
        words = []
        print(f'Speaker: {speakerID}')
        speaker = Speaker.speakerDict[speakerID]
        speechInstances = speaker.speechInstances
        if len(speechInstances) == 0:
            print('No speech to report.')
        for instance in speechInstances:
            words.append(instance.word)
        output = ' '.join(words)
        print(f'All words: {output}')
        print(f'WPM: {speaker.wpm}')


"""Analyze function will be used for NLP. Aim is to provide recommendations based on content. Stretch: fact check."""
def analyze(stream):
    t = 0
    while True:
        print(f'{t} seconds elapsed')
        t += 1
        time.sleep(1)


def enrollSpeakers():
    numSpeakers = input('Number Of Speakers: ')
    print('Select a training file for each speaker.')
    time.sleep(.5)
    gmms = []
    names = []
    # create gmms
    for i in range(int(numSpeakers)):
        filePath = utils.getAudioPath()
        name = input('Name of speaker: ')
        names.append(name)
        print(f'Processing {name} at {filePath}...')
        gmms.append(speakerIdentification.trainGMM(filePath))
        print(f'{filePath} processed.')
    # create speaker objects
    colors = utils.randomColors(len(names))
    for i, name in enumerate(names):
        Speaker(speakerId=name, color=colors[i])
    return gmms


def visualize(fig, axe, audioStream):
    patches = [mpatches.Patch(color=Speaker.speakerDict[speakerID].color,
                              label=f'{speakerID}')
               for speakerID in Speaker.speakerDict]
    while True:
        data = [np.max(part) for part in audioStream.storedRaw]
        fig.legend(handles=patches)
        axe.clear()
        axe.plot(data)
        for sectionStart in Conversation.orderedConversation:
            end = Conversation.orderedConversation[sectionStart][0]
            speakerID = Conversation.orderedConversation[sectionStart][1]
            color = Speaker.speakerDict[speakerID].color
            axe.axvspan(sectionStart, end + 1, color=color, alpha=0.5)
        fig.canvas.draw_idle()
        time.sleep(2)


def moderate():
    gmms = enrollSpeakers()
    print('Virtual Moderator Initiated!')
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE) as stream:
        stream.gmms = gmms
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        fig = plt.figure()  # the figure will be reused later
        fig.suptitle(f'Moderator Beta', fontsize=16)
        axe = fig.add_subplot(111)

        t1 = threading.Thread(target=transcribe, args=(responses, stream))
        t2 = threading.Thread(target=analyze, args=(stream,))
        t3 = threading.Thread(target=visualize, args=(fig, axe, stream))

        # start threads
        t1.start()
        t2.start()
        t3.start()

        plt.show()

        # wait until threads finish their job
        t1.join()
        t2.join()
        t3.join()


if __name__ == '__main__':
    moderate()
