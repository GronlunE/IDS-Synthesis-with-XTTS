"""
Created on 31.8. 2024

@author: GronlunE

Description:
This script performs text-to-speech (TTS) synthesis using a specified TTS model. It combines multiple reference audio files and text files to generate synthesized speech files.

The `synthesizer` function takes reference audio files, text files, and an output directory to generate synthesized audio files where each combination of reference audio and text is processed. A progress bar provides feedback on the processing status.

Usage:
- Ensure the TTS model is correctly installed and configured.
- Provide the directories containing reference audio files, text files, and specify an output directory.
- The script will read text files and generate corresponding audio files based on the reference audio.

Dependencies:
- TTS library for text-to-speech synthesis.
- Standard Python libraries (`os`, `glob`, `sys`, `tqdm`).
"""

from TTS.api import TTS
import os
import glob
import sys
from tqdm import tqdm

# Constants (Path Definitions)
REFERENCES_DIR = r"STAGE/synthesis/references"
TEXTS_DIR = r"STAGE/synthesis/texts"
OUTPUT_DIR = r"STAGE/synthesis/output"
FILE_TYPE = "*.wav"


class SuppressPrint:
    """
    Context manager to suppress print output temporarily.

    Usage:
    with SuppressPrint():
        # Code that produces unwanted output
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def synthesizer(reference_audio_dir, reference_texts_dir, output_dir, file_type = "*.wav"):
    """
    Synthesizes speech by combining reference audio files and text files.

    :param reference_audio_dir: Directory containing reference audio files.
    :param reference_texts_dir: Directory containing text files to be synthesized.
    :param output_dir: Directory where synthesized audio files will be saved.
    :param file_type: The file type of the reference audio files (default is "*.wav").
    :return: None
    """
    # Initialize TTS with your model and settings
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read all reference text files
    text_files = glob.glob(os.path.join(reference_texts_dir, '*.txt'))

    # Check if the reference_audio_dir is a directory
    if os.path.isdir(reference_audio_dir):
        # Use glob to find all .wav files in the directory
        reference_audios = glob.glob(os.path.join(reference_audio_dir, file_type))
    else:
        # If the path is neither a directory nor a .wav file, return an empty list
        reference_audios = []

    # Total combinations
    total_combinations = len(reference_audios) * len(text_files)

    # Initialize the progress bar
    with tqdm(total=total_combinations, desc="Processing", unit="file") as pbar:

        # Loop through each reference audio and text file
        for reference_audio in reference_audios:
            audio_name = os.path.splitext(os.path.basename(reference_audio))[0]

            for text_file in text_files:
                text_name = os.path.splitext(os.path.basename(text_file))[0]

                # Read the story text from the file
                with open(text_file, "r", encoding="utf-8") as f:
                    reference_text = f.read()

                # Output file name derived from input reference audio and text
                output_file = os.path.join(output_dir, f"xtts_{audio_name}_{text_name}.wav")

                # Update the progress bar with current audio and text file names
                pbar.set_postfix({"Reference": audio_name, "Text": text_name})

                # The block of code you want to measure
                with SuppressPrint():
                    tts.tts_to_file(text=reference_text,
                                    file_path=output_file,
                                    speaker_wav=reference_audio,
                                    language="en")

                pbar.update(1)


# Call the synthesizer function with the defined constants
synthesizer(REFERENCES_DIR, TEXTS_DIR, OUTPUT_DIR)
