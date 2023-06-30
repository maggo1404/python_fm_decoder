import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal

# 1. Eine 8-Bit-Stereo-WAV-Datei öffnen und als IQ-Stream lesen
sample_rate, stereo_data = wavfile.read('audio.wav')
# In ein eindimensionales Array komplexer Zahlen konvertieren
iq_stream = stereo_data[:, 0] + 1j * stereo_data[:, 1]

# 1.5. DC-Block
dc_block_alpha = 0.99
dc_block_output = signal.lfilter([1, -1], [1, -dc_block_alpha], iq_stream)
iq_stream = dc_block_output

# 2. IQ-Stream als Wasserfall ausgeben
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].specgram(iq_stream, Fs=sample_rate, cmap='jet')
axs[0, 0].set_title('Wasserfall-Darstellung des IQ-Streams')
axs[0, 0].set_xlabel('Zeit')
axs[0, 0].set_ylabel('Frequenz')

# 3. Sinus mit 625 kHz erstellen
duration = len(iq_stream) / sample_rate
time = np.linspace(0, duration, len(iq_stream))
frequency = 629000
sinusoid = np.exp(-1j * 2 * np.pi * frequency * time)  # Negatives Vorzeichen hinzugefügt

# 4. IQ-Stream und Sinus multiplizieren (mischen)
mixed_stream = iq_stream * sinusoid

# 5. Gemischten Stream als Wasserfall ausgeben
axs[0, 1].specgram(mixed_stream, Fs=sample_rate, cmap='jet')
axs[0, 1].set_title('Wasserfall-Darstellung des gemischten Streams')
axs[0, 1].set_xlabel('Zeit')
axs[0, 1].set_ylabel('Frequenz')

# 6. Den gemischten Stream mit einem Tiefpassfilter filtern
cutoff_freq = 5000
passband_width = 10000
stopband_attenuation = 60
nyquist_rate = sample_rate / 2
transition_width = (passband_width - cutoff_freq) / nyquist_rate

filter_length, beta = signal.kaiserord(stopband_attenuation, transition_width)
taps = signal.firwin(filter_length, cutoff_freq, window=('kaiser', beta), fs=sample_rate, pass_zero='lowpass')

filtered_stream = signal.lfilter(taps, 1.0, mixed_stream)

# 7. Gefilterten Stream als Wasserfall ausgeben
axs[0, 2].specgram(filtered_stream, Fs=sample_rate, cmap='jet')
axs[0, 2].set_title('Wasserfall-Darstellung des gefilterten Streams')
axs[0, 2].set_xlabel('Zeit')
axs[0, 2].set_ylabel('Frequenz')

# 8. Jeden 83. Sample in einen neuen Stream kopieren
decimation_factor = 166
decimated_stream = filtered_stream[::decimation_factor]

# Reduziere auch die Samplerate entsprechend
decimated_sample_rate = sample_rate // decimation_factor

# 9. Den kleinen Stream als Wasserfall ausgeben
axs[1, 0].specgram(decimated_stream, Fs=int(decimated_sample_rate), cmap='jet')
axs[1, 0].set_title('Wasserfall-Darstellung des decimierten Streams')
axs[1, 0].set_xlabel('Zeit')
axs[1, 0].set_ylabel('Frequenz')

# 10. Den decimierten Stream durch einen FM-Demodulator führen
demodulated_stream = np.angle(decimated_stream[1:] * np.conj(decimated_stream[:-1]))

# 11. Den demodulierten Stream als Wasserfall ausgeben
axs[1, 1].specgram(demodulated_stream, Fs=int(decimated_sample_rate), cmap='jet')
axs[1, 1].set_title('Wasserfall-Darstellung des demodulierten Streams')
axs[1, 1].set_xlabel('Zeit')
axs[1, 1].set_ylabel('Frequenz')

# 12. Den demodulierten Stream als Zeit/Spannungsdiagramm darstellen
time_demod = np.linspace(0, duration-decimation_factor/sample_rate, len(demodulated_stream))
axs[1, 2].plot(time_demod, demodulated_stream)
axs[1, 2].set_title('Demodulierter Stream (Zeit/Spannung)')
axs[1, 2].set_xlabel('Zeit')
axs[1, 2].set_ylabel('Spannung')

# Platz zwischen den Subplots erhöhen
plt.tight_layout()

# Alle Wasserfalldarstellungen gleichzeitig anzeigen
plt.show()

# Den demodulierten Stream als Audio ausgeben
wavfile.write('demodulated_audio.wav', int(decimated_sample_rate), demodulated_stream.astype(np.float32))
