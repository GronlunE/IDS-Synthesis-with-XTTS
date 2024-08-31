function [mean_duration, stdev_duration, filtered_durations] = get_syllable_duration_statistics(filepath)
    % Define the threshold for syllable segmentation
    threshold = 0.05;

    % Run syllable segmentation
    [~, bounds_t] = thetaseg({filepath}, threshold);

    % Compute syllable durations
    syllable_durations = computeSyllableDurations(bounds_t);
    
    % Extract durations from the cell array
    durations = syllable_durations{1}; % Assuming single file input

    % Check if durations are empty
    if isempty(durations)
        mean_duration = NaN;
        stdev_duration = NaN;
        return;
    end

    % Sort durations in ascending order
    sorted_durations = sort(durations);

    % Determine the index to exclude top 5% durations
    num_durations = length(sorted_durations);
    cutoff_index = floor(0.95 * num_durations);

    % Exclude the top 5% of the longest durations
    filtered_durations = sorted_durations(1:cutoff_index);

    % Calculate mean and standard deviation
    mean_duration = mean(filtered_durations);
    stdev_duration = std(filtered_durations);
end