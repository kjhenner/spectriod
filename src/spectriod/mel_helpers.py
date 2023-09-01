import librosa
import numpy as np
from PIL import Image


def audio_segment_iter(audio: np.ndarray, segment_size: int, pad: bool = False):
    i = 0
    while i < len(audio):
        audio_segment = audio[i:i+segment_size]
        if len(audio_segment) < segment_size:
            if pad:
                audio_segment = np.concatenate([
                    audio_segment,
                    np.zeros((segment_size - len(audio_segment), ))
                ])
                yield audio_segment
        else:
            yield audio_segment
        i += segment_size


def log_mel_to_audio(
        log_mel_spectrogram,
        sr=48000,
        hop_length=512,
        n_fft=2048,
        ref=1024
) -> np.ndarray:
    log_mel_spectrogram = librosa.db_to_power(log_mel_spectrogram, ref=ref)
    audio = librosa.feature.inverse.mel_to_audio(
        log_mel_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return audio


def audio_to_log_mel(
        audio: np.ndarray,
        y_res,
        top_db=80,
        sr=48000,
        hop_length=512,
        n_fft=2048
) -> np.ndarray:
    """Converts an np ndarray of raw audio to log scaled MEL spectrogram.

    Args:
        audio (np.ndarray): raw audio

    Returns:
        log_mel_spectrogram (np.ndarray): log scaled MEL spectrogram
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=y_res,
        fmax=sr/2
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=top_db)


def log_mel_spectrogram_to_image(
        log_mel_spectrogram: np.ndarray,
        top_db=80,
) -> Image.Image:
    """Converts an np ndarray log scaled MEL spectrogram to a PIL image.

    Args:
        log_mel_spectrogram (np.ndarray): log scaled MEL spectrogram

    Returns:
        image (PIL Image): grayscale image
    """
    bytedata = (
            (
                    (log_mel_spectrogram + top_db) * 255 / top_db
            ).clip(0, 255) + 0.5
    ).astype(np.uint8)
    return Image.frombytes(
        "L",
        (log_mel_spectrogram.shape[1], log_mel_spectrogram.shape[0]),
        bytedata.tobytes()
    )


def image_to_log_mel_spectrogram(
        image: Image,
        top_db=80,
) -> np.ndarray:
    """Converts a spectrogram PIL image to NumPy ndarray of raw audio.

    Args:
        image (PIL Image): grayscale image

    Returns:
        log_mel_spectrogram (np.ndarray): log scaled MEL spectrogram
    """
    bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
        (image.height, image.width))
    log_mel_spectrogram = (bytedata.astype("float") * top_db / 255) - top_db
    return log_mel_spectrogram


def audio_to_image(
        audio: np.ndarray,
        y_res,
        top_db=80,
        sr=48000,
        hop_length=512,
        n_fft=2048
) -> Image:
    """Converts a NumPy ndarray of raw audio to log scaled MEL spectrogram image.

    Args:
        audio (np.ndarray): raw audio

    Returns:
        image (PIL.Image): spectrogram image
    """
    log_mel_spectrogram = audio_to_log_mel(
        audio,
        y_res,
        top_db=top_db,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft
    )
    image = log_mel_spectrogram_to_image(log_mel_spectrogram, top_db=top_db)
    return image.crop((0, 0, image.width - 1, image.height))


def image_to_audio(
        image: Image,
        sample_rate=24000,
        top_db=80,
) -> np.ndarray:
    """Converts a spectrogram PIL image to NumPy ndarray of raw audio.

    Args:
        image (PIL Image): grayscale image

    Returns:
        audio (np.ndarray): raw audio
    """
    log_mel_spectrogram = image_to_log_mel_spectrogram(image, top_db=top_db)
    return log_mel_to_audio(log_mel_spectrogram, sr=sample_rate)