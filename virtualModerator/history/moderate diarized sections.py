import pyaudio
import numpy as np
import time
import matplotlib.pyplot as plt
from speechDiarization import extractFeature, clusterRealTime
import random
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
    def __init__(self):
        self.diarizedDict = dict()
        self.wordsDict = dict()
        self.conversationDiarized = dict()


class Speaker:
    speakerDict = dict()

    def __init__(self, speakerId):
        self.speakerID = speakerId
        self.speechInstances = []
        Speaker.speakerDict[speakerId] = self


class SpeechInstance:
    def __init__(self, start, end, words):
        self.start = start
        self.end = end
        self.words = words


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate):
        self.rate = self.chunk = rate
        # Create a thread-safe buffer of audio data
        self.buff = queue.Queue()
        self.closed = True
        self.storedFeatures = []

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
            self.storedFeatures.append(extractFeature(dataFloat))
            if chunk is None:
                return
            data = [utils.float2pcm(dataFloat)]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    dataFloat = np.frombuffer(chunk, np.float32)
                    self.storedFeatures.append(extractFeature(dataFloat))
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def randomColor(numSpeakers):
    distinctColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                      '#ffffff', '#000000']
    random.shuffle(distinctColors)
    return distinctColors[0:numSpeakers]


def visualize(audioMasterData, secondsMaster, diarizedDict, barColor, colors):
    patches = []
    plt.clf()
    for i, label in enumerate(diarizedDict):
        times = diarizedDict[label]
        color = colors[i]
        patches.append(mpatches.Patch(color=color, label=f'Spaker {i}'))
        for start,end in times:
            plt.axvspan(start, end+1, color=color, alpha=.5)
        plt.legend(handles=patches)
    plt.bar(secondsMaster, audioMasterData, color=barColor)
    plt.pause(.05)
    plt.draw()


def streamMic(numSpeakers, maxTime=120):
    # initialize mic
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    colors = randomColor(numSpeakers)
    # select distinct bar color
    while True:
        barColor = randomColor(1)[0]
        if barColor not in colors:
            break

    # initialize streaming variables
    keepGoing = True
    start = time.time()
    seconds = 0
    audioMasterData = []
    secondsMaster = []
    features = []
    diarizedDict = dict()

    # continues stream while less than max time
    while keepGoing:
        seconds += 1
        data = stream.read(CHUNK, seconds)
        audioData = np.frombuffer(data, np.float32)
        features.append(extractFeature(audioData, False, False, True))

        secondsMaster.append(seconds)
        audioMasterData.append(np.sum(audioData) / len(audioData))
        print(audioMasterData)
        visualize(audioMasterData, secondsMaster, diarizedDict, barColor, colors)
        if seconds % 20 == 0:
            secondsMaster = secondsMaster[::2]
            audioMasterData = audioMasterData[::2]
        if seconds % 5 == 0:
            _, diarizedDict = clusterRealTime(np.array(features), 2)
        # break loop after 60 seconds
        if time.time()-start>maxTime:
            print('Time Expired')
            keepGoing = False

    print('Shutting down')
    stream.close()
    audio.terminate()


def transcribe(responses, conversation):
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
                conversation.wordsDict[timedelta.total_seconds(start + end/2)] = word
            binWords(diarizedDict=conversation.diarizedDict,
                     wordsDict=conversation.wordsDict)
            printReport()
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0


def binWords(diarizedDict, wordsDict):
    for timeStamp in wordsDict:
        for speaker in diarizedDict:
            for start, end in diarizedDict[speaker]:
                if start <= timeStamp <= end:
                    speaker = Speaker.speakerDict[speaker]
                    for speechInstance in speaker.speechInstances:
                        speechSection = SpeechInstance(start, end, wordsDict[timeStamp])
                        speaker.speechInstances.append(speechSection)


def printReport():
    for speakerID in Speaker.speakerDict:
        print(f'Speaker: {speakerID}')
        speechInstances = Speaker.speakerDict[speakerID].speechInstances
        if len(speechInstances) == 0:
            print('No speech to report.')
        for instance in speechInstances:
            output = ''.join(instance.words)
            print(f'start: {instance.start} \t end: {instance.end} \t speech: {output}')


def analyze(stream, numSpeakers, conversation):
    # initialize viz. values
    colors = randomColor(numSpeakers)
    diarizedDict = dict()
    seconds = 0
    audioMasterData = []
    secondsMaster = []
    # select distinct bar color
    while True:
        barColor = randomColor(1)[0]
        if barColor not in colors:
            break
    while True:
        seconds += 1
        # visualize(audioMasterData, secondsMaster, diarizedDict, barColor, colors)
        if seconds % 5 == 0:
            _, diarizedDict = clusterRealTime(np.array(stream.storedFeatures), numSpeakers)
        conversation.diarizedDict = diarizedDict
        time.sleep(1)


def moderate(numSpeakers):
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag
    conversation = Conversation()
    client = speech.SpeechClient()

    # create speaker objects
    for n in range(numSpeakers):
        Speaker(n)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # # Now, put the transcription responses to use.
        # listenActLoop(responses, audioData)

        t1 = threading.Thread(target=transcribe, args=(responses, conversation))
        t2 = threading.Thread(target=analyze, args=(stream, numSpeakers, conversation))

        # start threads
        t1.start()
        t2.start()

        # wait until threads finish their job
        t1.join()
        t2.join()


if __name__ == '__main__':
    moderate(2)