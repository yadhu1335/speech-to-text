# import pyaudio
# import numpy as np
# from deepspeech import Model

# # Paths to your model and scorer
# # Paths to the model and scorer
# MODEL_PATH = "./models/deepspeech-0.9.3-models.pbmm"
# SCORER_PATH = "./models/deepspeech-0.9.3-models.scorer"

# # Audio recording settings
# RATE = 16000  # Sample rate (16kHz)
# CHUNK = 1024  # Audio chunk size
# FORMAT = pyaudio.paInt16  # 16-bit audio format
# CHANNELS = 1  # Mono audio

# # Initialize DeepSpeech model
# model = Model(MODEL_PATH)
# model.enableExternalScorer(SCORER_PATH)  # Enable scorer for improved accuracy

# # Initialize PyAudio for real-time audio capture
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True, frames_per_buffer=CHUNK)

# print("Recording... Speak into the microphone. Press Ctrl+C to stop.")

# # Initialize the streaming recognition
# ds_stream = model.createStream()

# try:
#     while True:
#         # Read audio data from the microphone
#         data = stream.read(CHUNK, exception_on_overflow=False)
#         audio_data = np.frombuffer(data, dtype=np.int16)

#         # Feed audio data into the DeepSpeech stream
#         ds_stream.feedAudioContent(audio_data)

#         # Get partial transcription (while you are speaking)
#         partial_text = ds_stream.intermediateDecode()
#         print(f"Partial transcription: {partial_text}", end="\r", flush=True)

# except KeyboardInterrupt:
#     # Stop recording on Ctrl+C
#     print("\nStopping...")

#     # Get the final transcription once you stop speaking
#     final_text = ds_stream.finishStream()
#     print(f"\nFinal transcription: {final_text}")

# finally:
#     # Clean up the resources
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()



# the below code is the same as the above. The below code is updated version to test the functionality of the scorer

import pyaudio
import numpy as np
from deepspeech import Model

# Paths to your model and scorer
MODEL_PATH = "./models/deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "./models/deepspeech-0.9.3-models.scorer"

# Audio recording settings
RATE = 16000  # Sample rate (16kHz)
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio

# Initialize DeepSpeech model
model = Model(MODEL_PATH)
model.enableExternalScorer(SCORER_PATH)  # Enable scorer for improved accuracy

# Initialize PyAudio for real-time audio capture
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... Speak into the microphone. Press Ctrl+C to stop.")

# Initialize the streaming recognition
ds_stream = model.createStream()

try:
    # List to store audio data for processing without the scorer later
    audio_frames = []

    while True:
        # Read audio data from the microphone
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_frames.append(audio_data)

        # Feed audio data into the DeepSpeech stream
        ds_stream.feedAudioContent(audio_data)

        # Get partial transcription (while you are speaking)
        partial_text = ds_stream.intermediateDecode()
        print(f"Partial transcription: {partial_text}", end="\r", flush=True)

except KeyboardInterrupt:
    # Stop recording on Ctrl+C
    print("\nStopping...")

    # Get the final transcription once you stop speaking
    final_text_with_scorer = ds_stream.finishStream()
    print(f"\nFinal transcription with scorer: {final_text_with_scorer}")

    # Concatenate all audio frames into a single array for reprocessing
    all_audio_data = np.hstack(audio_frames)

    # Transcribe the same audio without the scorer
    model.disableExternalScorer()  # Disable the scorer
    ds_stream_no_scorer = model.createStream()
    ds_stream_no_scorer.feedAudioContent(all_audio_data)
    final_text_without_scorer = ds_stream_no_scorer.finishStream()
    print(f"Final transcription without scorer: {final_text_without_scorer}")

finally:
    # Clean up the resources
    stream.stop_stream()
    stream.close()
    audio.terminate()
