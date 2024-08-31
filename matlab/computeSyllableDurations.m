function syllable_durations = computeSyllableDurations(bounds_t)
    % Initialize cell array to store syllable durations for each file
    syllable_durations = cell(length(bounds_t), 1);
    
    % Loop through each file
    for k = 1:length(bounds_t)
        % Extract boundaries in seconds
        boundaries = bounds_t{k};
        
        % Ensure boundaries are sorted
        boundaries = sort(boundaries);
        
        % Calculate durations by subtracting consecutive boundaries
        durations = diff(boundaries);
        
        % Store durations in the output cell array
        syllable_durations{k} = durations;
    end
end