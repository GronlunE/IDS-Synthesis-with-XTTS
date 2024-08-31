function [mean_duration, stdev_duration, filtered_durations] = get_syllable_duration_statistics(filepath)
    % Created on 31.8.2024
    % @author: GronlunE
    %
    % GET_SYLLABLE_DURATION_STATISTICS Calculate statistical measures of syllable durations.
    %
    % This function computes the mean and standard deviation of syllable durations
    % from a given audio file. It first performs syllable segmentation to obtain the
    % syllable boundaries, calculates syllable durations, filters out the top 5% of
    % longest durations, and then computes the mean and standard deviation of the
    % remaining durations.
    %
    % Input:
    %   filepath : string
    %       The path to the audio file from which syllable durations are to be computed.
    %
    % Output:
    %   mean_duration : double
    %       The mean duration of the syllables after filtering.
    %   stdev_duration : double
    %       The standard deviation of the syllable durations after filtering.
    %   filtered_durations : vector
    %       A vector containing the syllable durations after filtering out the top 5% longest durations.
    %
    % Example:
    %   [mean_duration, stdev_duration, filtered_durations] = get_syllable_duration_statistics('audio_file.wav');

    % Define the threshold for syllable segmentation
    threshold = 0.05;

    % Perform syllable segmentation on the audio file to obtain boundaries
    [~, bounds_t] = thetaseg({filepath}, threshold);

    % Compute syllable durations from the boundaries
    syllable_durations = computeSyllableDurations(bounds_t);

    % Extract the durations for the first file (assuming single file input)
    durations = syllable_durations{1};

    % Check if the durations vector is empty
    if isempty(durations)
        mean_duration = NaN;
        stdev_duration = NaN;
        filtered_durations = [];
        return; % Exit function if no durations are available
    end

    % Sort the durations in ascending order
    sorted_durations = sort(durations);

    % Determine the cutoff index to exclude the top 5% of longest durations
    num_durations = length(sorted_durations);
    cutoff_index = floor(0.95 * num_durations);

    % Filter out the top 5% longest durations
    filtered_durations = sorted_durations(1:cutoff_index);

    % Calculate the mean duration of the filtered durations
    mean_duration = mean(filtered_durations);

    % Calculate the standard deviation of the filtered durations
    stdev_duration = std(filtered_durations);
end
