import os
import librosa

import numpy as np
from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader


class MelSpeakerID(Dataset):
    def __init__(self, meta_file, root_dir):

        self.audio_path_and_speaker_id = self._get_metadata(meta_file, root_dir)

    def __getitem__(self, idx):
        wav, sr = librosa.core.load(self.audio_path_and_speaker_id[idx])

        wav = wav / np.abs(wav).max() * 0.999

        # already trimed silence

        constant_values = 0.
        out_dtype = np.float32

        # preemphasis
        wav = signal.lfilter([1], [1, -0.97], wav)

        # wave to mel
        D = librosa.stft(y=wav, n_fft=2048, hop_length=1100, win_length=275) #stft

        _mel_basis = librosa.filters.mel(sr, 2048, n_mels=80)
        mel = np.dot(_mel_basis, abs(D)) # linear to mel

        min_level = np.exp(-100 / 20 * np.log(10))
        S = 20 * np.log10(np.maximum(min_level, mel)) - 20

        S = np.clip((2 * 4) * ((S + 100) / (100)) - 4, -4, 4) # signal_normalization

        mel_spec = S.astype(np.float32)

        return mel_spec

    def __len__(self):
        pass

    def _get_metadata(self, meta_file, root_dir):
        with open(os.path.join(root_dir, meta_file), encoding='utf-8') as f:
            metadata = f.readlines()

        audio_paths = [os.path.join(root_dir, x.split('|')[0][2:]) for x in metadata]
        # text = [x.split('|')[1] for x in metadata]
        speaker_id = [x.split('|')[2] for x in metadata]

        metadata = list(zip(audio_paths, speaker_id))

        return metadata


if __name__ == '__main__':
    root_dir = '../datasets/NIKL_pre/'
    meta_file = 'metadata_train.csv'

    dataloader = MelSpeakerID(meta_file, root_dir)
