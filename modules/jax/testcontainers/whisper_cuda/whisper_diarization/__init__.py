import logging
from typing import Optional
from urllib.error import URLError

from core.testcontainers.core.container import DockerContainer
from core.testcontainers.core.waiting_utils import wait_container_is_ready


class JAXWhisperDiarizationContainer(DockerContainer):
    """
    JAX-Whisper-Diarization container for fast speech recognition, transcription, and speaker diarization.

    Example:

        .. doctest::

        >>> logging.basicConfig(level=logging.INFO)

        ... # You need to provide your Hugging Face token to use the pyannote.audio models
        >>> hf_token = "your_huggingface_token_here"

        >>> with JAXWhisperDiarizationContainer(hf_token=hf_token) as whisper_diarization:
        ... whisper_diarization.connect()
        ...
        ... # Example: Transcribe and diarize an audio file
        ... result = whisper_diarization.transcribe_and_diarize_file("/path/to/audio/file.wav")
        ... print(f"Transcription and Diarization: {result}")
        ...
        ... # Example: Transcribe and diarize a YouTube video
        ... result = whisper_diarization.transcribe_and_diarize_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        ... print(f"YouTube Transcription and Diarization: {result}")
    """

    def __init__(self, model_name: str = "openai/whisper-large-v2", hf_token: Optional[str] = None, **kwargs):
        super().__init__("nvcr.io/nvidia/jax:23.08-py3", **kwargs)
        self.model_name = model_name
        self.hf_token = hf_token
        self.with_env("NVIDIA_VISIBLE_DEVICES", "all")
        self.with_env("CUDA_VISIBLE_DEVICES", "all")
        self.with_kwargs(runtime="nvidia")  # Use NVIDIA runtime for GPU support
        self.start_timeout = 600  # 10 minutes
        self.connection_retries = 5
        self.connection_retry_delay = 10  # seconds

        # Install required dependencies
        self.with_command(
            "sh -c '"
            "pip install --no-cache-dir git+https://github.com/sanchit-gandhi/whisper-jax.git && "
            "pip install --no-cache-dir numpy soundfile youtube_dl transformers datasets pyannote.audio && "
            "python -m pip install --upgrade --no-cache-dir jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            "'"
        )

    @wait_container_is_ready(URLError)
    def _connect(self):
        # Check if JAX and other required libraries are properly installed and functioning
        result = self.run_command(
            "import jax; import whisper_jax; import pyannote.audio; "
            "print(f'JAX version: {jax.__version__}'); "
            "print(f'Whisper-JAX version: {whisper_jax.__version__}'); "
            "print(f'Pyannote Audio version: {pyannote.audio.__version__}'); "
            "print(f'Available devices: {jax.devices()}'); "
            "print(jax.numpy.add(1, 1))"
        )

        if "JAX version" in result.output.decode() and "Available devices" in result.output.decode():
            logging.info(f"JAX-Whisper-Diarization environment verified:\n{result.output.decode()}")
        else:
            raise Exception("JAX-Whisper-Diarization environment check failed")

    def connect(self):
        """
        Connect to the JAX-Whisper-Diarization container and ensure it's ready.
        This method verifies that JAX, Whisper-JAX, and Pyannote Audio are properly installed and functioning.
        It also checks for available devices, including GPUs if applicable.
        """
        self._connect()
        logging.info("Successfully connected to JAX-Whisper-Diarization container and verified the environment")

    def run_command(self, command: str):
        """
        Run a Python command inside the container.
        """
        exec_result = self.exec(f"python -c '{command}'")
        return exec_result

    def transcribe_and_diarize_file(
        self, file_path: str, task: str = "transcribe", return_timestamps: bool = True, group_by_speaker: bool = True
    ):
        """
        Transcribe and diarize an audio file using Whisper-JAX and pyannote.
        """
        command = f"""
import soundfile as sf
import torch
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from pyannote.audio import Pipeline
import numpy as np

def align(transcription, segments, group_by_speaker=True):
    transcription_split = transcription.split("\\n")
    transcript = []
    for chunk in transcription_split:
        start_end, text = chunk[1:].split("] ")
        start, end = start_end.split("->")
        start, end = float(start), float(end)
        transcript.append({{"timestamp": (start, end), "text": text}})

    new_segments = []
    prev_segment = segments[0]
    for i in range(1, len(segments)):
        cur_segment = segments[i]
        if cur_segment["label"] != prev_segment["label"]:
            new_segments.append({{
                "segment": {{"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]}},
                "speaker": prev_segment["label"]
            }})
            prev_segment = segments[i]
    new_segments.append({{
        "segment": {{"start": prev_segment["segment"]["start"], "end": segments[-1]["segment"]["end"]}},
        "speaker": prev_segment["label"]
    }})

    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    for segment in new_segments:
        end_time = segment["segment"]["end"]
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append({{
                "speaker": segment["speaker"],
                "text": " ".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1])
            }})
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({{"speaker": segment["speaker"], **transcript[i]}})

        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return segmented_preds

pipeline = FlaxWhisperPipline("{self.model_name}", dtype=jnp.bfloat16, batch_size=16)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="{self.hf_token}")

audio, sr = sf.read("{file_path}")
inputs = {{"array": audio, "sampling_rate": sr}}

# Transcribe
result = pipeline(inputs, task="{task}", return_timestamps={return_timestamps})

# Diarize
diarization = diarization_pipeline({{"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}})
segments = diarization.for_json()["content"]

# Align transcription and diarization
aligned_result = align(result["text"], segments, group_by_speaker={group_by_speaker})
print(aligned_result)
"""
        return self.run_command(command)

    def transcribe_and_diarize_youtube(
        self, youtube_url: str, task: str = "transcribe", return_timestamps: bool = True, group_by_speaker: bool = True
    ):
        """
        Transcribe and diarize a YouTube video using Whisper-JAX and pyannote.
        """
        command = f"""
import tempfile
import youtube_dl
import soundfile as sf
import torch
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from pyannote.audio import Pipeline
import numpy as np

def download_youtube_audio(youtube_url, output_file):
    ydl_opts = {{
        'format': 'bestaudio/best',
        'postprocessors': [{{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }}],
        'outtmpl': output_file,
    }}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def align(transcription, segments, group_by_speaker=True):
    transcription_split = transcription.split("\\n")
    transcript = []
    for chunk in transcription_split:
        start_end, text = chunk[1:].split("] ")
        start, end = start_end.split("->")
        start, end = float(start), float(end)
        transcript.append({{"timestamp": (start, end), "text": text}})

    new_segments = []
    prev_segment = segments[0]
    for i in range(1, len(segments)):
        cur_segment = segments[i]
        if cur_segment["label"] != prev_segment["label"]:
            new_segments.append({{
                "segment": {{"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]}},
                "speaker": prev_segment["label"]
            }})
            prev_segment = segments[i]
    new_segments.append({{
        "segment": {{"start": prev_segment["segment"]["start"], "end": segments[-1]["segment"]["end"]}},
        "speaker": prev_segment["label"]
    }})

    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    for segment in new_segments:
        end_time = segment["segment"]["end"]
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append({{
                "speaker": segment["speaker"],
                "text": " ".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1])
            }})
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({{"speaker": segment["speaker"], **transcript[i]}})

        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return segmented_preds

pipeline = FlaxWhisperPipline("{self.model_name}", dtype=jnp.bfloat16, batch_size=16)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="{self.hf_token}")

with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
    download_youtube_audio("{youtube_url}", temp_file.name)
    audio, sr = sf.read(temp_file.name)
    inputs = {{"array": audio, "sampling_rate": sr}}

    # Transcribe
    result = pipeline(inputs, task="{task}", return_timestamps={return_timestamps})

    # Diarize
    diarization = diarization_pipeline({{"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}})
    segments = diarization.for_json()["content"]

    # Align transcription and diarization
    aligned_result = align(result["text"], segments, group_by_speaker={group_by_speaker})
    print(aligned_result)
"""
        return self.run_command(command)

    def start(self):
        """
        Start the JAX-Whisper-Diarization container and wait for it to be ready.
        """
        super().start()
        self._wait_for_container_to_be_ready()
        logging.info("JAX-Whisper-Diarization container started and ready.")
        return self

    def _wait_for_container_to_be_ready(self):
        # Wait for a specific log message that indicates the container is ready
        self.wait_for_logs("Installation completed")

    def stop(self, force=True):
        """
        Stop the JAX-Whisper-Diarization container.
        """
        super().stop(force)
        logging.info("JAX-Whisper-Diarization container stopped.")

    @property
    def timeout(self):
        """
        Get the container start timeout.
        """
        return self.start_timeout
