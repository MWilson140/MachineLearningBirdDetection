import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio


class AudioTransform:
    def __init__(self, n_fft, hop_length, n_mels):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def __call__(self, audio_file):
        waveform, sr = torchaudio.load(audio_file)
        if sr != 44100:
            resample_transform = T.Resample(orig_freq=sr, new_freq=44100)
            waveform = resample_transform(waveform)
        waveform = waveform * (1 / torch.max(torch.abs(waveform)))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        mel_transform = T.MelSpectrogram(
            sample_rate=44100, 
            n_mels=self.n_mels, 
            win_length=self.n_fft, 
            hop_length=self.hop_length, 
            n_fft=self.n_fft
        )
        transformed = mel_transform(waveform)
        transformed = 10 * torch.log10(transformed + 1e-6)
        transformed = transformed.squeeze(0)
        transformed = transformed.permute(1, 0)  # Reordering dimensions: (time, mel)
        
        max_length = 128
        if transformed.shape[0] > max_length:
            transformed = transformed[:max_length, :]
        elif transformed.shape[0] < max_length:
            transformed = F.pad(transformed, (0, 0, 0, max_length - transformed.shape[0]))
        
        return transformed
