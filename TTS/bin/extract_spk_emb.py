import argparse
import sys
from argparse import RawTextHelpFormatter

import numpy as np
import torch
from tqdm import tqdm

from TTS.tts.utils.speakers import SpeakerManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute the accuracy of the encoder.\n\n"""
        """
        Example runs:
        python TTS/bin/eval_encoder.py emotion_encoder_model.pth emotion_encoder_config.json  dataset_config.json
        """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to model config file.",
    )

    parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
    parser.add_argument("--eval", type=bool, help="compute eval.", default=True)

    args = parser.parse_args()

    enc_manager = SpeakerManager(
        encoder_model_path=args.model_path, encoder_config_path=args.config_path, use_cuda=args.use_cuda
    )
    filelist = "/data/Speech-Backbones/Grad-TTS/resources/filelists/vctk/valid.txt"
    outpath = "/data/TTS/TTS/bin/valid_wav_to_spk_emb.pth"
    speaker_mapping = {}
    with open(filelist) as f:
        for line in tqdm(f):
            wav_path, text, spkid = line.split("|")
            embedd = enc_manager.compute_embedding_from_clip(wav_path)
            speaker_mapping[wav_path] = embedd

    with open(outpath, "wb") as f:
        torch.save(speaker_mapping, f)
