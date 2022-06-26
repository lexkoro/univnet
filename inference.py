import argparse
import glob
import os

import torch
import tqdm
from omegaconf import OmegaConf
from scipy.io.wavfile import write

from model.generator import Generator
from utils.stft import TacotronSTFT
from utils.utils import read_wav_np


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        hp = OmegaConf.create(checkpoint["hp_str"])

    stft = TacotronSTFT(
        hp.audio.filter_length,
        hp.audio.hop_length,
        hp.audio.win_length,
        hp.audio.n_mel_channels,
        hp.audio.sampling_rate,
        hp.audio.mel_fmin,
        hp.audio.mel_fmax,
        center=False,
    )

    sr, audio = read_wav_np(wavpath)

    model = Generator(hp).cuda()
    saved_state_dict = checkpoint["model_g"]
    new_state_dict = {}

    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict["module." + k]
        except:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval(inference=True)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, "*.mel"))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()

            if (
                args.output_folder is None
            ):  # if output folder is not defined, audio samples are saved in input folder
                out_path = melpath.replace(
                    ".mel", "_reconstructed_epoch%04d.wav" % checkpoint["epoch"]
                )
            else:
                basename = os.path.basename(melpath)
                basename = basename.replace(
                    ".mel", "_reconstructed_epoch%04d.wav" % checkpoint["epoch"]
                )
                out_path = os.path.join(args.output_folder, basename)
            write(out_path, hp.audio.sampling_rate, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="yaml file for config. will use hp_str from checkpoint if not given.",
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        type=str,
        required=True,
        help="path of checkpoint pt file for evaluation",
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="directory of mel-spectrograms to invert into raw audio.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="directory which generated raw audio is saved.",
    )
    args = parser.parse_args()

    main(args)
