import glob
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from librosa.util import normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchaudio import transforms
import math

from utils.stft import TacotronSTFT
from utils.utils import read_wav_np

MAX_WAV_VALUE = 32768.0
SEQ_LENGTH = int(1.0 * 44100)
MAX_SEQ_LENGTH = int(6.0 * 44100)
# NOISE_PATH = "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/noise_fullband"
RIR_PATH = "/home/alexander/Projekte/smallroom22050"
# glob.glob(
#     "/home/akorolev/master/projects/data/SpeechData/noise_data/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k/smallroom22050/**/*.wav",
#     recursive=True,
# )

MAX_WAV_VALUE = 32768.0


def load_w3_risen12(root_dir):
    items = []
    lang_dirs = os.listdir(root_dir)
    for d in lang_dirs:
        tmp_items = []
        speakers = []
        metadata = os.path.join(root_dir, d, "metadata.csv")
        with open(metadata, "r") as rf:
            for line in rf:
                cols = line.split("|")
                text = cols[1]
                if len(cols) < 3:
                    continue
                speaker = cols[2].replace("\n", "")
                wav_file = os.path.join(root_dir, d, "wavs", cols[0])

                if os.path.isfile(wav_file) and "ghost" not in wav_file.lower():
                    if MAX_SEQ_LENGTH > Path(wav_file).stat().st_size // 2 > SEQ_LENGTH:
                        sp_count = Counter(speakers)
                        if sp_count[speaker] < 500:
                            speakers.append(speaker)
                            tmp_items.append([wav_file, speaker])

        random.shuffle(tmp_items)
        speaker_count = Counter(speakers)
        for item in tmp_items:
            if speaker_count[item[1]] > 30:
                items.append(item[0])

    return items


def load_skyrim(root_dir):
    items = []
    speaker_dirs = os.listdir(root_dir)
    for d in speaker_dirs:
        wav_paths = glob.glob(os.path.join(root_dir, d, "*.wav"), recursive=True)
        wav_paths = [Path(x) for x in wav_paths if "ghost" not in x.lower()]
        np.random.shuffle(wav_paths)
        filtered_wav = [
            str(x)
            for x in wav_paths
            if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
        ]
        if len(filtered_wav) > 100:
            items.extend(filtered_wav[:400])
    print("Skyrim:", len(items))
    return items


def find_wav_files(data_path, is_gothic=False):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    if is_gothic:
        HERO_PATHS = [Path(x) for x in wav_paths if "pc_hero" in x.lower()]
        OTHER_PATHS = [Path(x) for x in wav_paths if "pc_hero" not in x.lower()]
        print(len(HERO_PATHS[:500]))
        np.random.shuffle(HERO_PATHS)
        wav_paths = OTHER_PATHS + HERO_PATHS[:500]
    else:
        wav_paths = [Path(x) for x in wav_paths if "ghost" not in x.lower()]
    filtered_wav = [
        str(x) for x in wav_paths if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
    ]
    return filtered_wav


def find_g2_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    HERO_PATHS = [Path(x) for x in wav_paths if "15" == x.lower().split("_")[-2]]
    OTHER_PATHS = [Path(x) for x in wav_paths if "15" != x.lower().split("_")[-2]]
    print("G2:", len(HERO_PATHS[:200]))
    np.random.shuffle(HERO_PATHS)
    wav_paths = OTHER_PATHS + HERO_PATHS[:200]

    filtered_wav = [
        str(x) for x in wav_paths if MAX_SEQ_LENGTH > x.stat().st_size // 2 > SEQ_LENGTH
    ]
    return filtered_wav


def custom_data_load(eval_split_size):
    gothic3_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/Gothic3",
        True,
    )
    print("G3: ", len(gothic3_wavs))
    risen1_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen1/")
    print("R1: ", len(risen1_wavs))
    risen2_wavs = load_w3_risen12("/home/alexander/Projekte/44k_SR_Data/Risen2/")
    print("R2: ", len(risen2_wavs))
    risen3_wavs = find_wav_files("/home/alexander/Projekte/44k_SR_Data/Risen3")
    print("R3: ", len(risen3_wavs))
    skyrim_wavs = load_skyrim("/home/alexander/Projekte/44k_SR_Data/Skyrim")
    print("Skyrim: ", len(skyrim_wavs))
    gothic2_wavs = find_g2_wav_files("/home/alexander/Projekte/44k_SR_Data/Gothic2")
    print("G2: ", len(gothic2_wavs))
    custom_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/CustomVoices",
        False,
    )
    vctk_wavs = find_wav_files(
        "/home/alexander/Projekte/44k_SR_Data/VCTK/wav44",
        False,
    )
    print("VCTK: ", len(vctk_wavs))

    wav_paths = (
        gothic2_wavs
        + gothic3_wavs
        + risen1_wavs
        + risen2_wavs
        + risen3_wavs
        + skyrim_wavs
        + custom_wavs
        + vctk_wavs
    )
    print("Train Samples: ", len(wav_paths))
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def create_dataloader(hp, args, train, device):
    valid_files, train_files = custom_data_load(20)
    if train:
        dataset = MelFromDisk(hp, train_files, args, train, device)
        return DataLoader(
            dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=False,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    else:
        dataset = MelFromDisk(hp, valid_files, args, train, device)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class MelFromDisk(Dataset):
    def __init__(self, hp, metadata_path, args, train, device):
        random.seed(hp.train.seed)
        self.hp = hp
        self.args = args
        self.train = train
        self.meta = metadata_path
        self.stft = TacotronSTFT(
            hp.audio.filter_length,
            hp.audio.hop_length,
            hp.audio.win_length,
            hp.audio.n_mel_channels,
            hp.audio.sampling_rate,
            hp.audio.mel_fmin,
            hp.audio.mel_fmax,
            center=False,
            device=device,
        )
        self.upsample = transforms.Resample(
            orig_freq=22050,
            new_freq=44100,
            resampling_method="kaiser_window",
            lowpass_filter_width=6,
            rolloff=0.99,
            dtype=torch.float32,
        )
        self.downsample = transforms.Resample(
            orig_freq=44100,
            new_freq=22050,
            resampling_method="kaiser_window",
            lowpass_filter_width=6,
            rolloff=0.99,
            dtype=torch.float32,
        )

        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.shuffle = hp.train.spk_balanced

        if train and hp.train.spk_balanced:
            # balanced sampling for each speaker
            speaker_counter = Counter((spk_id for audiopath, text, spk_id in self.meta))
            weights = [
                1.0 / speaker_counter[spk_id] for audiopath, text, spk_id in self.meta
            ]

            self.mapping_weights = torch.DoubleTensor(weights)

        elif train:
            weights = [1.0 / len(self.meta) for _ in self.meta]
            self.mapping_weights = torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = torch.multinomial(self.mapping_weights, 1).item()
            return self.my_getitem(idx)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping_weights)

    def my_getitem(self, idx):
        wavpath = self.meta[idx]
        sr, audio = read_wav_np(wavpath)

        audio = normalize(audio) * 0.95

        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(
                audio,
                (
                    0,
                    self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio),
                ),
                mode="constant",
                constant_values=0.0,
            )

        audio = torch.from_numpy(audio).unsqueeze(0)


        if random.random() < 0.1:
            # apply distortion to random samples with a 10% chance
            noise = torch.rand(size=(audio.shape[0],)) - 0.5  # get 0 centered noise
            speech_power = audio.norm(p=2)
            noise_power = noise.norm(p=2)
            scale = math.sqrt(math.e) * noise_power / speech_power  # signal to noise ratio of 5db
            audio = (scale * audio + noise) / 2

        downsampled = self.downsample(audio)
        audio = self.upsample(downsampled)

        mel = self.get_mel(wavpath, audio, sr)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length - 1
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio_len = self.hp.audio.segment_length
            audio = audio[:, audio_start : audio_start + audio_len]

        return mel, audio

    def get_mel(self, wavpath, audio, sr):
        # melpath = wavpath.replace(".wav", ".mel")
        # try:
        #     mel = torch.load(melpath, map_location="cpu")
        #     assert (
        #         mel.size(0) == self.hp.audio.n_mel_channels
        #     ), "Mel dimension mismatch: expected %d, got %d" % (
        #         self.hp.audio.n_mel_channels,
        #         mel.size(0),
        #     )

        # except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            # sr, wav = read_wav_np(wavpath)
        wav = audio.squeeze(0).cpu().float().numpy()
        assert (
            sr == self.hp.audio.sampling_rate
        ), "sample mismatch: expected %d, got %d at %s" % (
            self.hp.audio.sampling_rate,
            sr,
            wavpath,
        )

        if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            wav = np.pad(
                wav,
                (
                    0,
                    self.hp.audio.segment_length
                    + self.hp.audio.pad_short
                    - len(wav),
                ),
                mode="constant",
                constant_values=0.0,
            )

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = self.stft.mel_spectrogram(wav)

        mel = mel.squeeze(0)

    

        return mel
