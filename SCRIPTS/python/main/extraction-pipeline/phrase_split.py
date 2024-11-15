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
