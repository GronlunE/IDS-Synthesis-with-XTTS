"""

"""
import os
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import matlab.engine
from tqdm import tqdm


def get_f0_statistics(pitch):
    """

    :param pitch:
    :return:
    """
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    sd_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 100]  # Remove unvoiced parts
    perc_95 = np.percentile(f0_values, 95)
    perc_5 = np.percentile(f0_values, 5)
    range_f0 = perc_95 - perc_5
    return mean_f0, sd_f0, perc_95, perc_5, range_f0, f0_values


def get_spectral_tilt_statistics(y, pitch, sr):
    """

    :param pitch:
    :param sr:
    :return:
    """

    def calculate_spectral_tilt(frame):
        """

        :param frame:
        :return:
        """
        # Compute the magnitude spectrum
        magnitude = np.abs(frame)
        # Take the logarithm of the magnitude spectrum
        log_magnitude = np.log1p(magnitude)  # Using log1p for numerical stability
        # Perform linear regression to fit a line to the log-magnitude spectrum
        frequencies = np.arange(len(log_magnitude))
        coeffs = np.polyfit(frequencies, log_magnitude, 1)  # Fit a 1st degree polynomial
        spectral_tilt = coeffs[0]  # The slope of the fit
        return spectral_tilt

    # Get the pitch values and time stamps
    f0_values = pitch.selected_array['frequency']

    # Compute the Spectorgram
    n_fft = int(0.03 * sr)  # 30 ms window size
    hop_length = int(0.01 * sr)  # 10 ms step size
    Sxx = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(Sxx)

    # Adjust f0_values to match the length of the spectrogram's second dimension
    spectrogram_length = spectrogram.shape[1]

    if len(f0_values) < spectrogram_length:

        # Pad f0_values with zeroes if it is shorter
        f0_adjusted = np.pad(f0_values, (0, spectrogram_length - len(f0_values)), mode='constant')
    elif len(f0_values) > spectrogram_length:
        # Truncate f0_values if it is longer
        f0_adjusted = f0_values[:spectrogram_length]
    else:
        # No adjustment needed if lengths match
        f0_adjusted = f0_values

    # Compute the length difference
    length_diff = abs(len(f0_values) - spectrogram_length)
    filtered_spectrogram = spectrogram[:, f0_adjusted > 100]
    if length_diff > 50:
        print(f"Warning: Truncation/padding applied to f0_values. Amount truncated/padded: {spectrogram_length - len(f0_values)}")

    # Calculate spectral tilt for each frame
    spectral_tilts = np.array([calculate_spectral_tilt(frame) for frame in filtered_spectrogram.T])

    # Compute mean and standard deviation of spectral tilts
    mean_spectral_tilt = np.mean(spectral_tilts)
    sd_spectral_tilt = np.std(spectral_tilts)

    return mean_spectral_tilt, sd_spectral_tilt, spectral_tilts


def get_f0_delta_statistics(f0_values):
    """

    :param f0_values:
    :return:
    """
    if len(f0_values) < 2:
        return 0, 0  # Return zero for both mean and std deviation if not enough data

    tilt_delta_values = []
    for i in range(len(f0_values)):
        start_index = max(i - 10, 0)
        end_index = min(i + 10, len(f0_values))
        segment = f0_values[start_index:end_index]
        if len(segment) < 2:
            continue
        x = np.arange(len(segment))
        coef = np.polyfit(x, segment, 1)  # Linear fit (1st degree polynomial)
        tilt = coef[0]
        tilt_delta_values.append(tilt)

    if len(tilt_delta_values) == 0:
        return 0, 0

    mean_f0_delta = np.mean(tilt_delta_values)
    sd_f0_delta = np.std(tilt_delta_values)

    return mean_f0_delta, sd_f0_delta, tilt_delta_values


def get_syllable_duration_statistics(filepath):
    """

    :param filepath:
    :return:
    """
    # Matlab paths
    matlab_functions_base = r"matlab"
    thetaseg_path = r"matlab/thetaOscillator"
    gammatone_path = r"matlab/thetaOscillator/gammatone"

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Add the directory containing the MATLAB function to the MATLAB path
    eng.addpath(matlab_functions_base, nargout=0)
    eng.addpath(thetaseg_path, nargout=0)
    eng.addpath(gammatone_path, nargout=0)

    # Call the MATLAB function
    mean_duration, stdev_duration, syllable_durations = eng.get_syllable_duration_statistics(filepath, nargout=3)

    # Stop MATLAB engine
    eng.quit()

    return mean_duration, stdev_duration, syllable_durations


def calculate_measurements():
    """

    :return:
    """

    # Define directories and CSV filenames
    file_types = ["original", "enhanced", "denoised"]
    csv_names = ["synthesized.csv", "references.csv"]
    base_dir = r"synthesis_stage"
    output_dir = r"plot_data/scatter"

    # Initialize lists to store results
    results = {file_type: {csv_name: [] for csv_name in csv_names} for file_type in file_types}

    # Process files
    for file_type in file_types:
        for csv_name in csv_names:
            directory = os.path.join(base_dir, csv_name.replace('.csv', ''), file_type.lower())
            files = [file for file in os.listdir(directory) if file.endswith(".wav")]
            for file in tqdm(files, desc=f"Processing {file_type} - {csv_name}"):

                filepath = os.path.join(directory, file)

                target_sr = 16000  # Target sampling rate
                y, sr = librosa.load(filepath, sr=target_sr)
                snd = parselmouth.Sound(y, sr)
                pitch = call(snd, "To Pitch", 0.0, 75, 600)

                # F0
                mean_f0, sd_f0, perc_95, perc_5, range_f0, f0_values_for_kde = get_f0_statistics(pitch)

                # Tilt
                f0_values = pitch.selected_array['frequency']
                f0_values = f0_values[f0_values > 100]  # Remove unvoiced parts
                mean_f0_delta, sd_f0_delta = get_f0_delta_statistics(f0_values)

                # Syllable duration
                mean_syllable_duration, sd_syllable_duration, syllable_durations = get_syllable_duration_statistics(filepath)

                # Spectral tilt
                mean_spectral_tilt, sd_spectral_tilt, spectral_tilt_values = get_spectral_tilt_statistics(y, pitch, sr)

                # Collect results
                results[file_type][csv_name].append({
                    "file_name": file,
                    "f0_mean": mean_f0,
                    "f0_sd": sd_f0,
                    "f0_max": perc_95,
                    "f0_min": perc_5,
                    "f0_range": range_f0,
                    "f0_delta_mean": mean_f0_delta,
                    "f0_delta_sd": sd_f0_delta,
                    "spectral_tilt_mean": mean_spectral_tilt,
                    "spectral_tilt_sd": sd_spectral_tilt,
                    "syllable_duration_mean": mean_syllable_duration,
                    "syllable_duration_sd": sd_syllable_duration
                })

    # Save synthesised data to CSV files
    for file_type in file_types:
        for csv_name in csv_names:
            df = pd.DataFrame(results[file_type][csv_name])
            if csv_name == "synthesized.csv":
                filename = "xtts_" + file_type.lower() + ".csv"
            else:
                filename = file_type.lower() + ".csv"
            synthesized_output_path = os.path.join(output_dir, csv_name.replace('.csv', ''), file_type.lower(), filename)
            df.to_csv(synthesized_output_path, index=False)

    # Save references data CSV files
    for csv_name in csv_names:
        references_data = []
        for file_type in file_types:
            if csv_name in results[file_type]:
                df = pd.DataFrame(results[file_type][csv_name])
                df['File_Type'] = file_type
                references_data.append(df)

        if references_data:
            references_df = pd.concat(references_data, ignore_index=True)
            references_output_path = os.path.join(output_dir, csv_name)
            references_df.to_csv(references_output_path, index=False)

    print("Processing complete.")