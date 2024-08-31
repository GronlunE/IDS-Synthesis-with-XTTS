function syllable_durations = computeSyllableDurations(bounds_t)
    % Created on 31.8.2024
    % @author: GronlunE
    %
    % COMPUTESYLLABLEDURATIONS Calculate the durations of syllables from boundary times.
    %
    % This function takes a cell array of boundary times, where each cell contains
    % a vector of boundary times for a particular file. The function computes the
    % durations of syllables as the differences between consecutive boundary times.
    %
    % Input:
    %   bounds_t : cell array of vectors
    %       A cell array where each cell contains a vector of boundary times in seconds.
    %       Each vector represents the start and end times of syllables for a single file.
    %
    % Output:
    %   syllable_durations : cell array of vectors
    %       A cell array where each cell contains a vector of syllable durations in seconds.
    %       Each vector represents the durations of syllables for a single file.
    %
    % Example:
    %   bounds_t = {[0.5, 1.5, 2.5], [0.3, 1.2, 2.0]};
    %   durations = computeSyllableDurations(bounds_t);
    %   % durations will be {[1, 1], [0.9, 0.8]}.

    % Initialize cell array to store syllable durations for each file
    syllable_durations = cell(length(bounds_t), 1);

    % Loop through each file
    for k = 1:length(bounds_t)
        % Extract boundaries in seconds for the current file
        boundaries = bounds_t{k};

        % Ensure boundaries are sorted in ascending order
        boundaries = sort(boundaries);

        % Calculate durations by subtracting consecutive boundaries
        durations = diff(boundaries);

        % Store calculated durations in the output cell array
        syllable_durations{k} = durations;
    end
end
