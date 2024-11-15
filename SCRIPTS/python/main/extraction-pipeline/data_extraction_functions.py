"""
Created on 15.11.2024

@author: GronlunE

Description:
    This script provides functions for extraction and analyzing pitch (F0) and spectral characteristics, such as spectral tilt, F0 delta, and syllable durations. It provides various functions for extracting statistical features from audio files, including:

    - F0 statistics (pitch values above a threshold).
    - Spectral tilt calculation from the audio waveform and pitch.
    - F0 delta (change) statistics, including linear regression to compute the tilt.
    - Syllable duration statistics based on audio timing and pitch analysis.
    - Peak detection and change-based thresholding for analyzing F0 extrema.

    The functions are used for audio feature extraction, analysis of pitch contours, and determining changes in pitch over time for speech data.

Functions:
    - `get_f0_statistics(pitch)`: Extracts and returns F0 values above 100 Hz from a pitch object.
    - `get_spectral_tilt_statistics(y, pitch, sr)`: Calculates the spectral tilt for an audio signal `y` using its pitch and sample rate `sr`.
    - `get_f0_delta_statistics(pitch, window=5, step=1)`: Computes the mean and standard deviation of the tilt (slope) of log-transformed F0 values over a sliding window.
    - `get_syllable_duration_statistics(filepath, all_syldurs)`: Integrates syllable duration data with pitch analysis to determine valid syllable durations based on the F0 values.
    - `get_F0_delta_peakdet_thr_line(filepath, target_sr=16000)`: Computes F0 delta statistics based on the fitted line between extrema in the pitch contour.
    - `get_F0_delta_peakdet_thr_change(filepath, target_sr=16000, change_threshold=0.02)`: Performs F0 delta analysis using a change threshold to filter significant pitch changes, fitting lines between extrema.

Usage:
    - Input: Audio file(s) in `.wav` format, with appropriate sample rates (typically 16 kHz).
    - These functions extract various pitch-related features and store them as arrays or lists, which can be used for further analysis.
    - `get_syllable_duration_statistics` also requires syllable duration data (in `all_syldurs`), which maps syllables to their corresponding time boundaries.

Dependencies:
    - `parselmouth`: For pitch extraction using Praat.
    - `librosa`: For audio processing and pitch tracking (F0 extraction).
    - `numpy`: For numerical computations.
    - `scipy`: For signal processing (e.g., peak detection and linear regression).
    - Standard Python libraries (`os`).

Notes:
    - This script assumes that the input audio files are in `.wav` format, sampled at 16 kHz.
    - Syllable duration data (`all_syldurs`) is assumed to be pre-loaded and provided as a dictionary structure.
    - For `get_F0_delta_peakdet_thr_line` and `get_F0_delta_peakdet_thr_change`, change thresholds can be adjusted to filter extreme F0 shifts.

"""

import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
from scipy.signal import find_peaks
import os


def get_f0_statistics(pitch):
    """

    :param pitch:
    :return:
    """

    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 100]  # Remove unvoiced parts
    return f0_values


def get_spectral_tilt_statistics(y, pitch, sr):
    """

    :param pitch:
    :param sr:
    :return:
    """

    def calculate_spectral_tilt(frame):
        """Calculate the spectral tilt for a single frame."""

        # Take the logarithm of the magnitude spectrum
        log_magnitude = 20 * np.log(frame + 1e-10)

        # Check for NaN values
        nan_count = np.isnan(log_magnitude).sum()
        if nan_count > 0:
            print(f"NaN values detected in log_magnitude: {nan_count} NaN values")

        # Perform linear regression to fit a line to the log-magnitude spectrum
        frequencies = np.arange(len(log_magnitude))
        coeffs = np.polyfit(frequencies, log_magnitude, 1)  # Fit a 1st degree polynomial
        spectral_tilt = coeffs[0]  # The slope of the fit
        return spectral_tilt

    # Get the pitch values and time stamps
    f0_values = pitch.selected_array['frequency']

    # Compute the Spectrogram
    n_fft = int(0.03 * sr)  # 30 ms window size
    hop_length = int(0.01 * sr)  # 10 ms step size
    Sxx = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(Sxx)
    spectrogram_length = magnitude.shape[1]

    # Padding
    # Spectrogram is consistently 4 zeroes longer than F0.
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
    if length_diff > 50:
        print(f"Warning: Truncation/padding applied to f0_values. Amount truncated/padded: {spectrogram_length - len(f0_values)}")

    # Remove unvoiced parts of the spectrogram
    filtered_magnitude = magnitude[:, f0_adjusted > 100]

    # Calculate spectral tilt for each frame
    spectral_tilts = np.array([calculate_spectral_tilt(frame) for frame in filtered_magnitude.T])

    return spectral_tilts


def get_f0_delta_statistics(pitch, window=5, step=1):
    """
    Calculate the mean and standard deviation of the tilt of f0 values after logarithmic transformation.
    :param pitch: Pitch data structure containing f0 values.
    :param window: The window size for calculating the slope (tilt) of log F0 values.
    :param step: The step size for sliding the window through the F0 values.
    :return: mean_f0_delta, sd_f0_delta, delta_values
    """

    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 100]  # Filter out low f0 values (unvoiced regions)

    if len(f0_values) < 2:
        return 0, 0, []  # Return zeros and empty array if not enough data

    # Apply logarithmic transformation to f0 values
    log_f0_values = np.log(f0_values)

    # Check for NaN values
    nan_count = np.isnan(log_f0_values).sum()
    if nan_count > 0:
        print(f"NaN values detected in log_f0_values: {nan_count} NaN values")

    delta_values = []

    # Step through the F0 values with the specified step size
    for i in range(0, len(log_f0_values), step):
        # Define window
        start_index = max(i - window, 0)
        end_index = min(i + window, len(log_f0_values))

        segment = log_f0_values[start_index:end_index]
        if len(segment) < 2:
            continue

        # Linear fit to the segment
        x = np.arange(len(segment))
        coef = np.polyfit(x, segment, 1)  # Linear fit (1st degree polynomial)
        tilt = coef[0]  # The slope (tilt)

        # Append tilt (delta) value to the list
        delta_values.append(tilt)

    if len(delta_values) == 0:
        return 0, 0, []

    # Convert delta_values to a NumPy array
    delta_values = np.array(delta_values)

    return delta_values


def get_syllable_duration_statistics(filepath, all_syldurs):
    """
    Integrate MATLAB and Python to process syllable duration statistics.

    :param filepath: Path of the original file.
    :param all_syldurs: Dictionary containing syllable durations and bounds.
    :return: Array of valid syllable durations.
    """
    # Extract the base name to access the corresponding syllable data
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Get syllable durations and bounds_t from all_syldurs
    syllable_durations = all_syldurs["SYLDURS"][base_name]['syllable_durations']
    bounds_t = all_syldurs["SYLDURS"][base_name]['bounds_t']

    target_sr = 16000

    # Load audio file for pitch extraction
    y, sr = librosa.load(filepath, sr=target_sr)
    snd = parselmouth.Sound(y, sr)
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    time_stamps = pitch.ts()
    f0_values = pitch.selected_array['frequency']

    # Ensure syllable_durations is a list if it has only one element
    if isinstance(syllable_durations, float):
        syllable_durations = [syllable_durations]

    # Initialize valid syllable durations as a list
    valid_syllable_durations = []

    for j in range(len(syllable_durations)):
        syllable_start = bounds_t[j]
        syllable_end = bounds_t[j + 1]  # The end time is the start of the next bound

        # Find indices of time stamps that fall within the current syllable bounds
        indices_in_syllable = np.where((time_stamps >= syllable_start) & (time_stamps < syllable_end))[0]

        # Check if any F0 value in these indices is greater than 100 Hz
        if np.any(f0_values[indices_in_syllable] > 100):
            valid_syllable_durations.append(syllable_durations[j])

    # Convert valid syllable durations to a NumPy array if needed
    valid_syllable_durations = np.array(valid_syllable_durations)

    return valid_syllable_durations

    return valid_syllable_durations


def get_F0_delta_peakdet_thr_line(filepath, target_sr=16000):
    """
    Compute various statistics about the F0 contour including delta values from line fits between extrema.
    :param filepath: Path to the audio file.
    :param target_sr: Sample rate for loading the audio.
    :return: A tuple containing (log_abs_mean_f0_delta, log_abs_mean_f0_delta, abs_mean_f0_delta, abs_std_f0_delta, delta_values).
    """

    def filter_extrema(f0_values, baseline):
        """
        Filter extrema to select only the most extreme within each segment between two opposite types.
        :param f0_values: Array of f0 values.
        :param baseline: The baseline threshold value.
        :return: Filtered lists of maxima and minima.
        """

        # Find local maxima and minima
        maxima, _ = find_peaks(f0_values)
        minima, _ = find_peaks(-f0_values)

        valid_maxima = []
        valid_minima = []

        # Initialize segment tracking
        segment_extrema = []
        last_extrema_type = None

        for i in range(len(f0_values)):
            # Handle maxima
            if i in maxima:
                if f0_values[i] > baseline:
                    if last_extrema_type == 'max':
                        segment_extrema.append((i, f0_values[i], 'max'))
                    else:
                        # Process the previous segment
                        if last_extrema_type == 'min':
                            if segment_extrema:
                                if segment_extrema[0][2] == 'max':
                                    max_extreme = max(segment_extrema, key=lambda x: x[1])
                                    valid_maxima.append(max_extreme[0])
                                elif segment_extrema[0][2] == 'min':
                                    min_extreme = min(segment_extrema, key=lambda x: x[1])
                                    valid_minima.append(min_extreme[0])
                        # Start a new segment
                        segment_extrema = [(i, f0_values[i], 'max')]
                        last_extrema_type = 'max'

            # Handle minima
            elif i in minima:
                if f0_values[i] < baseline:
                    if last_extrema_type == 'min':
                        segment_extrema.append((i, f0_values[i], 'min'))
                    else:
                        # Process the previous segment
                        if last_extrema_type == 'max':
                            if segment_extrema:
                                if segment_extrema[0][2] == 'max':
                                    max_extreme = max(segment_extrema, key=lambda x: x[1])
                                    valid_maxima.append(max_extreme[0])
                                elif segment_extrema[0][2] == 'min':
                                    min_extreme = min(segment_extrema, key=lambda x: x[1])
                                    valid_minima.append(min_extreme[0])
                        # Start a new segment
                        segment_extrema = [(i, f0_values[i], 'min')]
                        last_extrema_type = 'min'

        # Handle the last segment
        if segment_extrema:
            if segment_extrema[0][2] == 'max':
                max_extreme = max(segment_extrema, key=lambda x: x[1])
                valid_maxima.append(max_extreme[0])
            elif segment_extrema[0][2] == 'min':
                min_extreme = min(segment_extrema, key=lambda x: x[1])
                valid_minima.append(min_extreme[0])

        return valid_maxima, valid_minima

    # Load the audio file
    y, sr = librosa.load(filepath, sr=target_sr)

    # Extract f0 contour using librosa's pitch tracking
    f0_values, voiced_flags, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr)

    # Remove unvoiced regions (NaNs)
    f0_values = f0_values[~np.isnan(f0_values)]
    f0_values = np.log(f0_values)

    # Compute the mean f0 to act as the baseline
    mean_f0 = np.mean(f0_values)

    # Filter extrema
    valid_maxima, valid_minima = filter_extrema(f0_values, mean_f0)

    # Prepare for fitting lines and calculating coefficients
    extrema_indices = sorted(valid_maxima + valid_minima)
    slopes = []

    for i in range(len(extrema_indices) - 1):
        start_idx = extrema_indices[i]
        end_idx = extrema_indices[i + 1]

        # Ensure we are connecting maxima and minima
        if (start_idx in valid_maxima and end_idx in valid_minima) or (
                start_idx in valid_minima and end_idx in valid_maxima):
            x_values = np.array([start_idx, end_idx])
            y_values = np.array([f0_values[start_idx], f0_values[end_idx]])
            coeffs = np.polyfit(x_values, y_values, 1)
            slopes.append(coeffs[0])

    return slopes


def get_F0_delta_peakdet_thr_change(filepath, target_sr=16000, change_threshold=0.02):
    """
    Calculate delta F0 metrics and return key statistics based on the fitted line slopes.
    :param filepath: Path to the audio file.
    :param target_sr: Target sample rate for loading the audio file.
    :param change_threshold: Minimum F0 change to consider an extreme.
    :return: log_abs_mean_f0_delta, log_abs_std_f0_delta, abs_mean_f0_delta, abs_std_f0_delta, delta_values
    """

    # Load the audio file
    y, sr = librosa.load(filepath, sr=target_sr)

    # Extract f0 contour using librosa's pitch tracking
    f0_values, voiced_flags, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr)

    # Remove unvoiced regions (NaNs)
    f0_values = f0_values[~np.isnan(f0_values)]
    f0_values = np.log(f0_values)

    # Find local maxima and minima
    maxima, _ = find_peaks(f0_values)
    minima, _ = find_peaks(-f0_values)

    valid_maxima = []
    valid_minima = []

    # Apply change threshold and filter out insignificant extrema
    for i in range(1, len(maxima)):
        if abs(f0_values[maxima[i]] - f0_values[maxima[i - 1]]) >= change_threshold:
            valid_maxima.append(maxima[i])

    for i in range(1, len(minima)):
        if abs(f0_values[minima[i]] - f0_values[minima[i - 1]]) >= change_threshold:
            valid_minima.append(minima[i])

    # Ensure that maxima and minima alternate
    extrema_indices = sorted(valid_maxima + valid_minima)
    slopes = []

    for i in range(len(extrema_indices) - 1):
        start_idx = extrema_indices[i]
        end_idx = extrema_indices[i + 1]

        # Ensure we are connecting maxima and minima
        if (start_idx in valid_maxima and end_idx in valid_minima) or (
                start_idx in valid_minima and end_idx in valid_maxima):
            x_values = np.array([start_idx, end_idx])
            y_values = np.array([f0_values[start_idx], f0_values[end_idx]])
            coeffs = np.polyfit(x_values, y_values, 1)
            slopes.append(coeffs[0])

    return slopes
