import os
import librosa

import numpy as np
from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader


class MelSpeakerID(Dataset):
    def __init__(self, meta_file, root_dir, speaker_id_list=None):
        self.speaker_id_list = speaker_id_list
        self.audio_path_and_speaker_id = self._get_metadata(meta_file, root_dir)

    def __getitem__(self, idx):
        wav, sr = librosa.core.load(self.audio_path_and_speaker_id[idx][0])
        speaker_id = self.audio_path_and_speaker_id[idx][1]

        wav = wav / np.abs(wav).max() * 0.999

        # already trimed silence

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

        return torch.from_numpy(mel_spec), int(speaker_id.strip())

    def __len__(self):
        return len(self.audio_path_and_speaker_id)

    def _get_metadata(self, meta_file, root_dir):
        if self.speaker_id_list is None:
            with open(os.path.join(root_dir, meta_file), encoding='utf-8') as f:
                metadata = f.readlines()

            audio_paths = [os.path.join(root_dir, x.split('|')[0][2:]) for x in metadata]
            # text = [x.split('|')[1] for x in metadata]
            speaker_id = [x.split('|')[2] for x in metadata]

            processed_metadata = list(zip(audio_paths, speaker_id))
        else:
            sp_idxs = self.speaker_id_list

            with open(os.path.join(root_dir, meta_file), encoding='utf-8') as f:
                metadata = f.readlines()

            processed_metadata = []
            for i in range(len(metadata)):
                m = metadata[i].split('|')

                if int(m[2].strip()) in sp_idxs:
                    # tmp.append((os.path.join(root_dir, m[0][2:]), m[2]))
                    processed_metadata.append((os.path.join(root_dir, m[0][2:]), str(sp_idxs.index(int(m[2].strip())))))

        return processed_metadata

class MelCollate():
    def __init__(self):
        pass
    def __call__(self, batch):

        # include mel padded
        mel_padded = torch.zeros(len(batch), 80, 236)
        mel_padded -= 4

        for i in range(len(batch)):
            mel = batch[i][0]
            mel_padded[i, :, :mel.size(1)] = mel

        return mel_padded.unsqueeze(1), [x[1] for x in batch]


if __name__ == '__main__':
    root_dir = '../datasets/NIKL_pre/'
    meta_file = 'metadata_train.csv'
    dataloader = MelSpeakerID(meta_file, root_dir)
    collate_fn = MelCollate()

    dataloader = DataLoader(dataloader, batch_size=4,
                            shuffle=True, num_workers=1, collate_fn=collate_fn)

    for i_batch, x in enumerate(dataloader):
        print(x[0].shape)
        break
