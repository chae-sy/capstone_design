import torchaudio


def sp_transform(waveform):
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=1024, win_length=1024, hop_length=555, n_mels=10
    )

    spectrogram = spectrogram_transform(waveform)
    epsilon = 1e-6
    spectrogram = spectrogram + epsilon
    spectrogram = spectrogram.log2()

    return spectrogram
