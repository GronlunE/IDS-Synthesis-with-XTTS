% Define directories
REF_PHRASE_DIR = 'G:\Research\XTTS_Test\DATA\IDS-ADS\phrases\references';
SYNTH_PHRASE_DIR = 'G:\Research\XTTS_Test\DATA\IDS-ADS\phrases\syntheses';
SUB_DIRS = {'original', 'denoised', 'enhanced'};
MAT_OUTPUT_DIR = 'G:\Research\XTTS_Test\DATA\IDS-ADS';

% Initialize cell arrays to store file names
all_filenames = {};

% Collect WAV files from reference phrase directories
for i = 1:length(SUB_DIRS)
    dir_path = fullfile(REF_PHRASE_DIR, SUB_DIRS{i});
    files = dir(fullfile(dir_path, '*.wav'));
    for j = 1:length(files)
        all_filenames{end+1} = fullfile(dir_path, files(j).name);
    end
end

% Collect WAV files from synthesis phrase directories
for i = 1:length(SUB_DIRS)
    dir_path = fullfile(SYNTH_PHRASE_DIR, SUB_DIRS{i});
    files = dir(fullfile(dir_path, '*.wav'));
    for j = 1:length(files)
        all_filenames{end+1} = fullfile(dir_path, files(j).name);
    end
end

% Define threshold for syllable segmentation
threshold = 0.05;

% Run thetaseg on all collected files
[bounds, bounds_t, osc_env, nuclei] = thetaseg(all_filenames, threshold);
%% 
% Prepare syllable durations
all_syllable_durations = cell(length(bounds_t), 1);

% Create a waitbar for processing syllable durations
hWaitbar = waitbar(0, 'Computing syllable durations...', 'Name', 'Syllable Duration Progress');

startTime = tic;  % Start the timer

for k = 1:length(bounds_t)
    syllable_durations = computeSyllableDurations(bounds_t{k});
    all_syllable_durations{k} = syllable_durations;
    
    % Update waitbar with estimated time remaining
    elapsedTime = toc(startTime);
    estimatedTime = (elapsedTime / k) * (length(bounds_t) - k);
    waitbar(k / length(bounds_t), hWaitbar, ...
        sprintf('Computing syllable durations for file %d of %d...\nEstimated time remaining: %.1f seconds', k, length(bounds_t), estimatedTime));
end

% Close the waitbar
close(hWaitbar);
%% 

for k = 1:length(all_filenames)
    [~, base_name, ~] = fileparts(all_filenames{k});
    
    % Replace spaces with underscores in base_name
    base_name_clean = strrep(base_name, ' ', '_');
    
    % Create a new field in SYLDURS using the cleaned base name
    SYLDURS.(base_name_clean).syllable_durations = all_syllable_durations{k};
    SYLDURS.(base_name_clean).bounds_t = bounds_t{k}; 
end

% Save results to a MAT file under the key 'SYLDURS'
output_file = fullfile(MAT_OUTPUT_DIR, 'IDS-ADS_syllable_durations_all.mat');
var SYLDURS;
save(output_file, 'SYLDURS');

disp('Syllable durations have been computed and saved successfully.');

