import os
import csv
import shutil  # Import shutil for file copying
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

# Define file paths
tsv_file = r"E:\COMMONVOICE\validated.tsv"
clips_dir = r"E:\COMMONVOICE\clips"
output_dir = r"G:\Research\XTTS_Test\DATA\.audio\.commonvoice2"
speaker_id_file = r"G:\Research\XTTS_Test\DATA\.audio\.commonvoice\speaker_id.txt"


# Function to calculate SNR
def calculate_snr(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    signal_power = np.mean(samples ** 2)

    # Estimate noise as the mean of the quietest samples (e.g., lowest 10% of amplitudes)
    noise_power = np.mean(np.sort(samples)[:int(len(samples) * 0.1)] ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    return snr


# Step 1: Load existing speaker IDs from the speaker_id_file
existing_speakers = set()
with open(speaker_id_file, 'r') as f:
    for line in f:
        # Extract speaker IDs from the file
        speaker_id = line.strip().split('=')[1].strip()  # Assuming format: speaker_n = client_id
        existing_speakers.add(speaker_id)

# Step 2: Collect all audio files and their sizes with tqdm
file_sizes = []
all_files = [filename for filename in os.listdir(clips_dir) if filename.endswith('.mp3')]  # Filter files first

# Using tqdm to show progress while collecting file sizes
for filename in tqdm(all_files, desc="Collecting file sizes", unit="file"):
    full_path = os.path.join(clips_dir, filename)
    file_size = os.path.getsize(full_path)  # Get file size in bytes
    file_sizes.append((filename, full_path, file_size))

# Step 3: Sort files by size and select the top 10%
file_sizes.sort(key=lambda x: x[2], reverse=True)  # Sort by file size (largest to smallest)

# Get the number of files to keep (top 10%)
num_files_to_keep = max(1, len(file_sizes) // 10)  # Ensure at least one file is selected
top_files = file_sizes[:num_files_to_keep]  # Select top 10% of files

# Step 4: Create a mapping of client_ids from the TSV file
client_id_mapping = {}
with open(tsv_file, 'r', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')

    # Create a client_id mapping
    for row in tqdm(reader, desc="Mapping client_ids", total=len(top_files), unit="file"):
        client_id = row['client_id']
        file_name = row['path'] + ".mp3"
        client_id_mapping[file_name] = client_id  # Create a mapping of file names to client IDs

# Step 5: Calculate SNR for top files
snrs = []  # Initialize a list to hold SNR values

for filename, full_path, file_size in tqdm(top_files, desc="Calculating SNR", unit="file"):
    audio_segment = AudioSegment.from_mp3(full_path)
    snr = calculate_snr(audio_segment)
    snrs.append(snr)  # Collect SNR for later analysis

snrs = snrs[np.isfinite(snrs)]

# Step 6: Collect clips based on length and SNR criteria
selected_clips = {}

for filename, full_path, file_size in tqdm(top_files, desc="Selecting unique clips", unit="file"):
    client_id = client_id_mapping.get(filename)

    if client_id and client_id not in existing_speakers:
        audio_segment = AudioSegment.from_mp3(full_path)
        clip_length = audio_segment.duration_seconds  # Get clip length in seconds

        # Check if the clip meets the length and SNR criteria
        if 10 < clip_length <= 25 and snr > (max(snrs) - min(snrs)) / 2 + min(snrs):  # Length and SNR criteria
            selected_clips[client_id] = full_path  # Keep only one clip per unique speaker
            if len(selected_clips) >= 100:  # Stop when we have 100 unique speakers
                break

# Calculate best and worst SNR from selected clips
best_snr = max(snrs) if snrs else float('-inf')
worst_snr = min(snrs) if snrs else float('inf')

print(f"Best SNR: {best_snr} dB")
print(f"Worst SNR: {worst_snr} dB")

# Ensure we have exactly 100 clips; fill if necessary
if len(selected_clips) < 100:
    for filename, full_path, file_size in tqdm(top_files, desc="Filling selection", unit="file"):
        client_id = client_id_mapping.get(filename)

        if client_id not in selected_clips:
            audio_segment = AudioSegment.from_mp3(full_path)
            clip_length = audio_segment.duration_seconds  # Get clip length in seconds

            # Check if the clip meets the length criteria only to fill the selection
            if 10 < clip_length <= 25:
                selected_clips[client_id] = full_path
                if len(selected_clips) >= 100:
                    break

# Step 7: Save the selected clips to the output directory and create a speaker list
client_ids = []
os.makedirs(output_dir, exist_ok=True)

for idx, (client_id, clip_path) in enumerate(selected_clips.items(), start=6):
    output_clip_path = os.path.join(output_dir, f"speaker_{idx}.mp3")
    shutil.copy(clip_path, output_clip_path)  # Copy the file to the output directory
    client_ids.append(f"speaker_{idx} = {client_id}")

# Step 8: Write the client_ids to the text file in the output directory
with open(os.path.join(output_dir, 'speaker_ids.txt'), 'w') as f:
    for client_id_line in client_ids:
        f.write(client_id_line + '\n')

print(f"Selected {len(selected_clips)} clips and saved to {output_dir}")
