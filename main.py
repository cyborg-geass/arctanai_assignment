# Import necessary Libraries
import pyaudio
import wave
import numpy as np
from df.enhance import enhance, init_df, resample
from pyannote.audio import Pipeline
import warnings
import os
import torch

warnings.filterwarnings('ignore')

# Constants for audio streaming
CHUNK = 1024  # audio samples per chunk
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio for single microphone need to change for multi-microphones but mostly it is single channel
RATE = 16000  # Sampling rate in Hz it needs to be resampled in real time
LATENCY = 0.1  # Target latency in seconds (100 milisconds)
OUTPUT_FILE = "output_real_time.wav"

# Initialize the DeepFilterNet model
model, df_state, _ = init_df()

# Initialize Pyannote audio pipeline for speaker diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
).to(torch.device("cpu")) # use cuda for better & faster results

# def single_speaker_scenario():
#     pass
# def multi_speaker_scenario():
#     pass

# Identifying Dominant speaker from multi speakers when the mode is single; so we can 
# only get the one speaker in multi-speaker scenario
def identify_dominant_speaker_by_energy(audio_data, sample_rate):
    temp_file = "temp_audio.wav" # Saving the live streaming audio data in a temporary file
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1) # Setting channel as 1; can do with 2 or 3 based on the microphones or input devices
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate) # saving the frame rate as RATE
        wf.writeframes(audio_data.astype(np.int16).tobytes()) # Saving the frames

    diarization = pipeline(temp_file) # Diarization
    speaker_energies = {} # Finding out argmax speaker energy through the intensity or energy

    for segment, _, speaker in diarization.itertracks(yield_label=True): # Iterable through the track
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        segment_energy = np.sum(audio_data[start:end] ** 2) 
        speaker_energies[speaker] = speaker_energies.get(speaker, 0) + segment_energy

    os.remove(temp_file)  # Cleaning up temporary file

    if not speaker_energies: # If there is no sound or blank
        raise ValueError("No speakers detected in the audio stream.")

    dominant_speaker = max(speaker_energies, key=speaker_energies.get)
    return dominant_speaker, diarization


def real_time_denoising(mode="single"):
    """Real-time noise cancellation with single or multi-speaker mode."""
    p = pyaudio.PyAudio()
    # Initialzing the stream
    stream_in = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Real-time noise cancellation system started...")

    frames = []

    try:
        while True: # The stream will go on until execution stopped
            data = stream_in.read(CHUNK, exception_on_overflow=False) # taking the data from stream
            audio_data = np.frombuffer(data, dtype=np.int16)
            # In two modes we have to take single and multi speakers
            ## If the mode is single & there are multiple modes we need to eliminate other non-dominant speakers
            if mode == "single":
                try:
                    print("Applying single-speaker noise cancellation...")
                    dominant_speaker, diarization = identify_dominant_speaker_by_energy(audio_data, RATE)

                    retained_audio = np.zeros_like(audio_data)
                    for segment, _, speaker in diarization.itertracks(yield_label=True):
                        if speaker == dominant_speaker:
                            start = int(segment.start * RATE)
                            end = int(segment.end * RATE)
                            retained_audio[start:end] = audio_data[start:end]

                    audio_tensor = resample(
                        torch.Tensor(np.array([retained_audio])), RATE, df_state.sr(), method="sinc_best"
                    ) # Also can use 'sinc_fast' as well for method
                    enhanced_audio = enhance(model, df_state, audio_tensor)
                except ValueError as e: # If there are empty chunks or any value errors
                    print(f"Warning: {e}. Skipping this chunk.")
                    continue

            ## We only need to remove background noise & all the speakers will be there in the recording
            elif mode == "multi":
                print("Applying multi-speaker noise cancellation...")
                audio_tensor = resample(
                    torch.Tensor(np.array([audio_data])), RATE, df_state.sr(), method="sinc_best"
                ) # Also can use 'sinc_fast' as well for method 
                enhanced_audio = enhance(model, df_state, audio_tensor)

            else: # Choose between 'single' and 'multi'
                raise ValueError("Invalid mode. Choose 'single' or 'multi'.")

            enhanced_audio_resampled = resample(
                enhanced_audio, df_state.sr(), RATE, method="sinc_best"
            ).numpy()

            frames.append(enhanced_audio_resampled.flatten())

    except KeyboardInterrupt:
        print("Stopping real-time noise cancellation...")
    finally:
        stream_in.stop_stream()
        stream_in.close()
        p.terminate()

        if frames:
            with wave.open(OUTPUT_FILE, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(RATE)
                wf.writeframes(np.concatenate(frames).astype(np.int16).tobytes())

            print(f"Processed audio saved to {OUTPUT_FILE}")
        else: ## If there are no frames no output file will be processed
            print("No audio processed; output file not created.")


if __name__ == "__main__": ## Start the stream from here
    mode = input("Enter mode (single/multi): ").strip().lower()
    real_time_denoising(mode=mode)
