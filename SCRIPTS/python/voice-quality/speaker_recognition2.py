import os
import pandas as pd
from speechbrain.inference import SpeakerRecognition
from tqdm import tqdm
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
base_audio_path = r"G:\Research\XTTS_Test\DATA\IDS-with-XTTS\syntheses\enhanced"
validation_dir = r"G:\Research\XTTS_Test\DATA\.audio\speaker_verification_validation"
commonvoice_dir = r"G:\Research\XTTS_Test\DATA\.audio\.commonvoice2"
csv_file_path = r"G:\Research\XTTS_Test\CODE\python\speaker_verification.csv"

# DataFrame to store the results
results_df = pd.DataFrame(columns=["file1", "file2", "score", "prediction", "comparison_type"])

# Load CSV file
csv_df = pd.read_csv(csv_file_path)

# Get all unique file names from file1 and file2 columns
unique_files = pd.unique(csv_df[['file1', 'file2']].values.ravel('K'))

# Step 1: Collect the relevant files from syntheses and speaker_verification_validation
syntheses_files = [f for f in unique_files if f.endswith('.wav') and os.path.exists(os.path.join(base_audio_path, f))]
validation_files = [f for f in unique_files if f.endswith('.mp3') and os.path.exists(os.path.join(validation_dir, f))]

# Step 2: Load files from .commonvoice2
commonvoice_files = [f for f in os.listdir(commonvoice_dir) if f.endswith(('.wav', '.mp3'))]
commonvoice_files.sort()  # Ensure they're sorted in a consistent order

# Step 3: Precompile all file comparison pairs (120x100 = 12,000 pairs)
comparisons = []

# Add pairs of syntheses_files and validation_files against commonvoice_files
for base_file in syntheses_files + validation_files:
    base_file_path = os.path.join(base_audio_path if base_file in syntheses_files else validation_dir, base_file)
    for commonvoice_file in commonvoice_files:
        commonvoice_file_path = os.path.join(commonvoice_dir, commonvoice_file)
        comparisons.append((base_file_path, commonvoice_file_path))

# Step 4: Define a helper function for verification
def verify_and_store(file1, file2, comparison_type):
    """Performs speaker verification between two files and stores the result."""
    score, prediction = verification.verify_files(file1, file2)
    # Convert tensors into float and boolean respectively
    score = float(score.item())  # Extract the float value from tensor
    prediction = bool(prediction.item())  # Convert tensor to boolean
    return {"file1": os.path.basename(file1), "file2": os.path.basename(file2), "score": score, "prediction": prediction, "comparison_type": comparison_type}

# Step 5: Run verification for each pair in the comparisons array
for file1, file2 in tqdm(comparisons, desc="Verifying all pairs"):
    result = verify_and_store(file1, file2, "cross_comparison")
    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

# Step 6: Save results to a new CSV
results_df.to_csv(r"G:\Research\XTTS_Test\CODE\python\speaker_verification_2.csv", index=False)
