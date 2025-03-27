import tensorflow as tf
import matplotlib.pyplot as plt

# Charger l'audio
audio_file = 'wavs/segment_10.wav'
audio = tf.io.read_file(audio_file)
audio, _ = tf.audio.decode_wav(audio)

# Si l'audio est stéréo, convertir en mono
audio = tf.reduce_mean(audio, axis=-1)

# Calculer le spectrogramme
spectrogram = tf.signal.stft(
    audio, frame_length=256, frame_step=128, fft_length=256
)

# Convertir en dB
spectrogram_db = 10 * tf.math.log(tf.abs(spectrogram) + 1e-6) / tf.math.log(10.0)

# Afficher le spectrogramme
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram_db.numpy().T, aspect='auto', origin='lower', cmap='inferno')
plt.title('Spectrogram')
plt.xlabel('Temps')
plt.ylabel('Fréquence')
plt.colorbar(label='Intensité (dB)')
plt.savefig('spectrogram.png')  # Sauvegarde l'image sous 'spectrogram.png'
 