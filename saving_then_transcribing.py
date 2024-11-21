import wave
import os
import pyaudio
import numpy as np
from deepspeech import Model

# Paths to the model and scorer
MODEL_PATH = "./models/deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "./models/deepspeech-0.9.3-models.scorer"

# Audio recording settings
RATE = 16000  # Sample rate (16kHz)
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
TEMP_WAV_PATH = "./temp_audio/temp_audio.wav"  # Path for temporary audio file

# Initialize DeepSpeech model
model = Model(MODEL_PATH)
model.enableExternalScorer(SCORER_PATH)

# Initialize PyAudio for real-time audio capture
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... Speak into the microphone. Press Ctrl+C to stop.")

# Record audio
frames = []

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

except KeyboardInterrupt:
    print("\nStopping recording...")

    # Save the recorded audio to a temporary file
    print(f"Saving audio to {TEMP_WAV_PATH}...")
    with wave.open(TEMP_WAV_PATH, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Audio saved. Starting transcription...")

    # Transcribe the audio file
    with wave.open(TEMP_WAV_PATH, 'rb') as wf:
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        transcription = model.stt(audio_data)

    print(f"Transcription: {transcription}")

finally:
    # Clean up the resources
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Optionally remove the temporary file
    if os.path.exists(TEMP_WAV_PATH):
        os.remove(TEMP_WAV_PATH)
        print(f"Temporary file {TEMP_WAV_PATH} deleted.")
