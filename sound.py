import sounddevice as sd
import numpy as np
import wavio
import librosa
import matplotlib.pyplot as plt
import librosa.display

#we explicitly declare the duration time of the recording to be saved and the sampling frequency
duration = 5
fs = 10000  

#  function records the audio using sounddevice module imported above
def record_audio(duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording finished.")
    return recording

#function saves the audio to the file using wavio 
def save_audio_to_file(filename, audio_data, fs):
    audio_data_normalized = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, audio_data_normalized, fs, sampwidth=2)

#functiom makes use of matplotlib.pyplot to plot the required spectrograms in realtime(with slight delay ofcourse)
def plot_spectrogram(y, sr, hop_length, y_axis="linear"):   
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(y, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    global spectrogram_2d_array
    spectrogram_2d_array = np.array(librosa.amplitude_to_db(y, ref=np.max))
    print(spectrogram_2d_array)
    plt.show()

if __name__ == "__main__":
    output_filename = "recorded_audio.wav"

    recorded_audio = record_audio(duration, fs)

    save_audio_to_file(output_filename, recorded_audio, fs)

    # save the audio file using librosa
    audio_file = output_filename
    y, sr = librosa.load(audio_file, sr=None)  # sr=None keeps the native sampling rate

    # STFT
    n_fft = 4096
    hop_length = 1024
    s_scale = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Get the magnitude spectrogram
    y_scale = np.abs(s_scale)**2

    # Plot the spectrogram
    plot_spectrogram(y_scale, sr, hop_length)

print(np.max(spectrogram_2d_array))