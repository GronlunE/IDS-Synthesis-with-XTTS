"""

"""
from TTS.api import TTS
import os
import glob
import sys
from tqdm import tqdm


class SuppressPrint:
    """

    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def synthesizer(reference_audio_dir, reference_texts_dir, output_dir, file_type = "*.wav"):
    """

    :param reference_audio_dir:
    :param reference_texts_dir:
    :param output_dir:
    :param file_type:
    :return:
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


references = r"synthesis_stage/references"
texts = r"synthesis_stage/texts"
output = r"synthesis_stage/synthesized"
file_type = "*.wav"

synthesizer(references, texts, output)
