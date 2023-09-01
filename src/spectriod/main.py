import wave

from queue import Queue
from threading import Thread

from mel_helpers import image_to_audio, audio_to_image
import cv2
import PIL
import numpy as np
import pyaudio
import soundfile as sf
from scipy.io import wavfile

SPECTROGRAM_SHAPE = (513, 256)


class AudioPlayer:
    chunk = 1024

    def __init__(self, queue):
        """ Init audio stream """
        self.queue = queue
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True
        )

    def start(self):
        """ Play entire file """
        while True:
            audio_data = self.queue.get()

            # If the producer puts None in the queue, the player will terminate
            if audio_data is None:
                break

            wavfile.write('test.wav', 48000, audio_data)
            wf = wave.open('test.wav', 'rb')
            data = wf.readframes(self.chunk)
            while data != b'':
                self.stream.write(data)
                data = wf.readframes(self.chunk)
            wf.close()

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)  # 0 is your default webcam
    queue = Queue(maxsize=1)  # only allows one item in the queue at a time for flow control

    player = AudioPlayer(queue)
    player_thread = Thread(target=player.start)
    player_thread.start()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        image = PIL.Image.fromarray(frame).convert("L").resize(SPECTROGRAM_SHAPE)
        audio = image_to_audio(image)
        audio_int16 = np.int16(audio * 32767)
        queue.put(audio_int16)

        # Display the resulting frame
        cv2.imshow('Your streaming video', np.array(image))

        # exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    queue.put(None)  # tells the player to stop
    player_thread.join()
    cap.release()
    cv2.destroyAllWindows()