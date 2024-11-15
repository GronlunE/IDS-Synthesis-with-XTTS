import os
import csv
from tqdm import tqdm
from pydub import AudioSegment

# File paths
tsv_file = r"E:\COMMONVOICE\validated.tsv"
clips_dir = r"E:\COMMONVOICE\clips"
pause_file = r"G:\Research\XTTS_Test\CODE\python\pause.wav"
output_dir = r"G:\Research\XTTS_Test\DATA\.audio\.commonvoice"
speaker_id_file = r"G:\Research\XTTS_Test\DATA\.audio\.commonvoice\speaker_id.txt"


# Function to load audio clips based on client_id and concatenate them into 20s+ segments
def concatenate_clips(tsv_file, clips_dir, pause_file, speaker_id_file):
    client_clips = {}  # Dictionary to store client_id and their respective audio segments

    # Count total rows in the TSV file first
    with open(tsv_file, 'r', encoding='utf-8') as tsvfile:
        total_rows = sum(1 for _ in tsvfile) - 1  # Subtract 1 for the header row

    # Read the TSV file to organize clips by client_id
    with open(tsv_file, 'r', encoding='utf-8') as tsvfile:

        reader = csv.DictReader(tsvfile, delimiter='\t')

        # Load clips into client_clips dictionary
        for row in tqdm(reader, desc="Loading clips", total=total_rows, unit="file"):
            client_id = row['client_id']  # Ensure this column exists in your TSV
            file_name = row['path'] + ".mp3"
            file_path = os.path.join(clips_dir, file_name)

            if os.path.exists(file_path):
                if client_id in client_clips:
                    client_clips[client_id].append(file_path)  # Store the path
                else:
                    client_clips[client_id] = [file_path]  # Initialize with a list

    # Sort clients by the number of clips they have (descending)
    sorted_clients = sorted(client_clips.items(), key=lambda item: len(item[1]), reverse=True)

    # Select the top 5 clients with the most clips
    top_5_clients = sorted_clients[:5]

    # Create pause audio segment
    pause_audio = AudioSegment.from_wav(pause_file)

    # Create or overwrite speaker_id.txt to store speaker-to-client mapping
    with open(speaker_id_file, 'w') as speaker_file:
        # Process each of the top 5 clients
        for client_idx, (client_id, clips) in enumerate(
                tqdm(top_5_clients, desc="Processing top 5 clients", unit="speaker"), start=1):
            segment_count = 0
            current_segment = AudioSegment.silent(duration=0)  # Start with silence
            current_duration = 0  # Track current concatenated duration for each segment

            # Write speaker_n = client_id mapping in speaker_id.txt
            speaker_file.write(f"speaker_{client_idx} = {client_id}\n")

            for clip_path in clips:
                clip = AudioSegment.from_mp3(clip_path)  # Load the audio file
                current_segment += clip + pause_audio
                current_duration += len(clip + pause_audio)

                # If the duration exceeds 20 seconds, export the segment
                if current_duration >= 20000:  # 20000 ms = 20 seconds
                    segment_count += 1
                    output_file_path = os.path.join(output_dir,
                                                    f"verification_speaker_{client_idx}_concat_{segment_count}.mp3")
                    current_segment.export(output_file_path, format="mp3")

                    # Reset for the next segment
                    current_segment = AudioSegment.silent(duration=0)  # Start fresh for the next segment
                    current_duration = 0

                    # Stop after 4 segments for this client
                    if segment_count == 4:
                        break

            # If there are fewer than 4 segments, export what remains (even if it's less than 20 seconds)
            if segment_count < 4 and len(current_segment) > 0:
                remaining_segment_path = os.path.join(output_dir,
                                                      f"verification_speaker_{client_idx}_concat_{segment_count + 1}.mp3")
                current_segment.export(remaining_segment_path, format="mp3")


# Run the modified concatenation function and create speaker_id.txt
concatenate_clips(tsv_file, clips_dir, pause_file, speaker_id_file)
