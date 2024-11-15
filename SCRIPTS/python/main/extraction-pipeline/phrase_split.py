"""
Created on 15.11.2024

@author: GronlunE

Description:
    This script processes TextGrid and audio files to extract specific audio segments based on time intervals defined in the TextGrid.
    It focuses on the 'Phrases' tier within the TextGrid files, extracting intervals where non-empty text is present, and uses these intervals
    to segment corresponding audio files into smaller clips. The script saves the extracted audio clips as individual WAV files with filenames
    based on the phrase index. It processes all TextGrid files in a given root directory, looks for matching audio files in a specified
    directory, and saves the resulting phrase audio files in a structured output directory.

    The `parse_textgrid` function parses TextGrid files to extract time intervals and text from the 'Phrases' tier, ensuring only
    non-empty text intervals are returned. The `extract_audio_segments` function extracts audio segments based on the parsed
    intervals and exports them to the specified output directory. The `process_files` function iterates through all TextGrid files
    in the specified directory, extracts audio segments, and organizes them into output directories based on the type of the source
    file (e.g., 'references' or 'syntheses').

Usage:
    - Ensure the TextGrid files and corresponding audio files are located in the specified directories.
    - Set the `textgrid_root_dir` and `audio_root_dir` to the appropriate directories for your TextGrid and audio files.
    - Set the `output_base_dir` to the desired directory where phrase audio files will be saved.
    - The script will process all TextGrid files, extract the audio segments based on the defined intervals, and save the audio clips
      into structured subdirectories under `output_base_dir`.

Dependencies:
    - `os` for file and directory operations.
    - `re` for regular expression handling to parse TextGrid files.
    - `glob` for file searching and pattern matching.
    - `pydub` for audio processing and exporting audio segments.
    - `tqdm` for progress bar visualization during processing.

Notes:
    - This script assumes that the TextGrid files have a `.auto.TextGrid` extension and that corresponding audio files are in WAV format.
    - The output directory structure will be based on the source file type (`references` or `syntheses`) and category (subdirectory name).
    - Ensure that both TextGrid and audio files are organized in the same relative structure for correct matching.
    - If no valid intervals are found in a TextGrid file, that file is skipped.
    - The script handles the extraction and saving of phrases only when matching audio files are found.

"""


import os
import re
import glob
from pydub import AudioSegment
from tqdm import tqdm


def parse_textgrid(file_path):
    """Parse the TextGrid file to extract intervals from the 'Phrases' tier with non-empty text."""
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expression to locate the "Phrases" tier and its intervals
    tier_pattern = re.compile(
        r'item \[\d+\]:\s*class = "IntervalTier"\s*name = "Phrases"\s*xmin = [\d\.]+\s*xmax = [\d\.]+\s*intervals: size = \d+\s*(.*?)\s*item \[\d+\]:',
        re.DOTALL)
    tier_match = tier_pattern.search(data)

    if tier_match:
        # Extract intervals inside the 'Phrases' tier
        tier_data = tier_match.group(1)

        # Regular expression to match interval details (xmin, xmax, text)
        interval_pattern = re.compile(r'xmin = ([\d\.]+)\s*xmax = ([\d\.]+)\s*text = "(.*?)"')
        intervals = interval_pattern.findall(tier_data)

        # Only return intervals where the text is not empty
        phrases = [(float(xmin), float(xmax), text) for xmin, xmax, text in intervals if text.strip()]
        return phrases
    else:
        # Return empty list if "Phrases" tier is not found
        return []


def extract_audio_segments(audio_file, phrases, output_dir, base_filename):
    """Extract audio segments from the WAV file based on the phrases' time intervals."""
    audio = AudioSegment.from_wav(audio_file)

    # Use tqdm to track the progress of extracting each phrase
    for idx, (xmin, xmax, text) in enumerate(
            tqdm(phrases, desc=f"Extracting phrases from {base_filename}", unit="phrase"), 1):
        start_time = int(xmin * 1000)  # Convert seconds to milliseconds
        end_time = int(xmax * 1000)

        segment = audio[start_time:end_time]

        # Create the output filename
        output_filename = f"{base_filename}_phrase_{idx}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Export the audio segment
        segment.export(output_path, format="wav")


def process_files(textgrid_root_dir, audio_root_dir, output_base_dir):
    """Process all TextGrid files in the root directory and save phrase audio files."""
    textgrid_pattern = os.path.join(textgrid_root_dir, "**", "*.auto.TextGrid")

    textgrid_files = glob.glob(textgrid_pattern, recursive=True)

    # Use tqdm to track the progress of processing each TextGrid file
    for textgrid_file in tqdm(textgrid_files, desc=f"Processing TextGrid files in {textgrid_root_dir}", unit="file"):
        base_dir = os.path.dirname(textgrid_file)
        base_filename = os.path.splitext(os.path.basename(textgrid_file))[0].replace(".auto", "")

        # Parse the TextGrid file for phrases from the 'Phrases' tier
        phrases = parse_textgrid(textgrid_file)

        if not phrases:
            continue

        # Locate the corresponding audio file (assumed .wav with the same name in the audio_root_dir)
        audio_file = os.path.join(audio_root_dir, base_dir.replace(textgrid_root_dir, '').strip(os.sep),
                                  f"{base_filename}.wav")

        if os.path.exists(audio_file):
            # Determine whether the file comes from "references" or "synthesized"
            if "references" in textgrid_file:
                file_type = "references"
            elif "syntheses" in textgrid_file:
                file_type = "syntheses"
            else:
                continue

            # Extract the category (subdirectory name)
            category = os.path.basename(os.path.dirname(textgrid_file))

            # Define the output directory structure
            output_dir = os.path.join(output_base_dir, file_type, category)
            os.makedirs(output_dir, exist_ok=True)

            # Extract and save the audio segments
            extract_audio_segments(audio_file, phrases, output_dir, base_filename)


# Main entry point
if __name__ == "__main__":

    # Define the root directories containing the TextGrid files and audio files
    textgrid_root_dir = "G:/Research/XTTS_Test/DATA/IDS-with-XTTS/text_grids_praat"
    audio_root_dir = "G:/Research/XTTS_Test/DATA/IDS-with-XTTS"

    # Define the base output directory for phrase files
    output_base_dir = "G:/Research/XTTS_Test/DATA/IDS-with-XTTS/phrases"

    # Process all the TextGrid files and their corresponding audio files
    process_files(textgrid_root_dir, audio_root_dir, output_base_dir)
