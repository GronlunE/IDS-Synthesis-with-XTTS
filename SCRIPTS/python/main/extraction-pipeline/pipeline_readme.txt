PIPELINE STRUCTURE:

First must have phrases for each file. 
1. Download and use Praat script syllablenuclei_v3 for your syntheses and references and extract the textgrids
   into DATA/main/phrases/text-grids.

2. Use phrase_split.py and adjust the variables "textgrid_root_dir", "audio_root_dir" and "output_base_dir" to be appropriate for you.

3. Extract syllable durations from the phrases. -> refer to SCRIPTS/matlab/main/syllabledurations_readme.txt

3. collect_data.py collect() Here also change the directory variables to appropriate ones.

4. stat_into_csv Again adjust the directory variables

