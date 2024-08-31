# IDS-Synthesis-with-XTTS
Analyzing the IDS (Infant-Directed Speech) cloning capabilities of a TTS model

## Installation:
### Start here

1. Clone the repository and install [thetaOScillator](https://github.com/orasanen/thetaOscillator/blob/master/thetaseg.m) into the "thetaOscillator" folder and follow the instructions provided there for installation. The MATLAB version was used here. The environment variable does not contain TTS.
2. ->

### ONLY EXTRACT DATA ->

1. From the provided .zip file, extract the contents of the references into the "references" folder, synthesised audio into the "synthesised" folder, and texts into the "texts" folder under "synthesis_stage".
2. Run `statistics.py`
3. Run `display.py`

### Synthesize with the SAME references as I ->

1. From the provided .zip file, extract the contents of the references into the "references" folder and texts into the "texts" folder under "synthesis_stage".
2. Install [Coqui TTS](https://github.com/coqui-ai/TTS) by following their installation instructions. The "XTTS" folder can be used to store the model(s).
3. Run `synthesizer.py`
4. Run `statistics.py`
5. Run `display.py`

### Synthesize with DIFFERENT references as I ->

1. Extract your references into the "original" folder under "synthesis_stage/references" and texts into the "texts" folder under "synthesis_stage".
2. Install [Coqui TTS](https://github.com/coqui-ai/TTS) by following their installation instructions. The "XTTS" folder can be used to store the model(s).
3. Run `synthesizer.py`
4. ????

## Project Description

The project aims to use an XTTS model to synthesize Infant-Directed Speech (IDS) from IDS references, even though the model was not specifically designed for this purpose. Due to the poor quality of the IDS references, the model from [Resemble AI](https://github.com/resemble-ai/resemble-enhance) was employed to clean up the audio using their "denoise" and "enhance" functionalities.

Subsequently, prosodic and quality features were extracted from both the synthesized and reference audio. The extracted statistics were then visualized using various types of figures, including scatter plots with ellipse fitting to mean and standard deviation, Kernel Density Estimates (KDEs) from density functions, and radar plots summarizing all statistics together. Comparisons were made between the original IDS reference and the syntheses of all original, denoised, and enhanced versions.

## Examples

### References:

- Original IDS:
- Enhanced IDS:
- Denoised IDS:

### Syntheses:

- Original IDS:
- Enhanced IDS:
- Denoised IDS:

### Figures

- Scatter:
- KDE of Density:
- Radar:
