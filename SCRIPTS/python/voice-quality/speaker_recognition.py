import os
import random
from itertools import combinations
from speechbrain.inference import SpeakerRecognition
from tqdm import tqdm
import pandas as pd
import logging
import warnings

# Set logging level for speechbrain to WARNING
logging.getLogger("speechbrain").setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the SpeakerRecognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Paths
base_audio_path = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\syntheses"
validation_dir = r"G:\Research\XTTS_Test\DATA\.audio\speaker_verification_validation"

# DataFrame to store the results
results_df = pd.DataFrame(columns=["file1", "file2", "score", "prediction", "comparison_type"])


# Helper function to perform the verification and store results
def verify_and_store(file1, file2, comparison_type):
    score, prediction = verification.verify_files(file1, file2)

    # Extract file names from the paths (remove paths)
    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)

    # Convert tensors into float and boolean respectively
    score = float(score.item())  # Extract the float value from tensor
    prediction = bool(prediction.item())  # Convert tensor to boolean

    return {"file1": file1_name, "file2": file2_name, "score": score, "prediction": prediction,
            "comparison_type": comparison_type}


# Step 1: Compare within the base_audio_path
category = 'enhanced'
category_path = os.path.join(base_audio_path, category)
audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
audio_files.sort()

# Step 1.1: Create a dictionary for i values (1 to 50) and j values (1 to 10)
i_dict = {}
num_i_values = 50
j_range = range(1, 11)  # j values from 1 to 10

# Populate the dictionary
for i in range(1, num_i_values + 1):
    j_values = random.sample(j_range, 2)  # Pick 2 unique random numbers from j_range
    i_dict[i] = j_values  # Store in the dictionary

# Step 1.2: Generate a list of audio file names based on the i_dict
file_list = []
for i, j_values in i_dict.items():
    for j in j_values:
        filename = f'xtts_enhanced_concat_{i}_GILES_{j}.wav'
        if filename in audio_files:
            file_list.append(filename)

# Step 1.3: Generate combinations of the audio files
file_combinations = list(combinations(file_list, 2))

# Step 1.4: Verify and store results for each combination
for file1, file2 in tqdm(file_combinations, desc="Verifying internal base files"):
    result = verify_and_store(os.path.join(category_path, file1), os.path.join(category_path, file2), "internal_base")
    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

# Step 2: Compare all files in the validation directory
validation_files = [f for f in os.listdir(validation_dir) if f.endswith('.mp3')]
validation_files.sort()
validation_combinations = list(combinations(validation_files, 2))

# Compare all combinations in validation directory with tqdm
for file1, file2 in tqdm(validation_combinations, desc="Validation pairs"):
    result = verify_and_store(os.path.join(validation_dir, file1), os.path.join(validation_dir, file2),
                              "internal_validation")
    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

# Step 3: Cross-compare each selected base file with all validation files
for base_file in tqdm(file_list, desc="Cross-comparing with validation files"):
    for validation_file in tqdm(validation_files, desc="Comparing with each validation file", leave=False):
        result = verify_and_store(os.path.join(category_path, base_file), os.path.join(validation_dir, validation_file),
                                  "cross_base_validation")
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

# Save the DataFrame to CSV, ensuring paths are removed and tensors are converted
results_df.to_csv("speaker_verification_results_fixed.csv", index=False)

# Step 4: Calculate Equal Error Rate (EER) (To be done later based on scores and predictions)
# Save the necessary scores and predictions for EER analysis in the CSV as well
