# IDS-Synthesis-with-XTTS
Analyzing the IDS (Infant-Directed Speech) synthesizing capabilities of a TTS model

## Installation:
### Start here

1. Clone the repository and install [thetaOscillator](https://github.com/orasanen/thetaOscillator/blob/master) into the "thetaOscillator" folder and follow the instructions provided there for installation. The MATLAB version was used here. The environment variable does not contain TTS.
2. Matlabengine is required to one function to use [thetaOscillator](https://github.com/orasanen/thetaOscillator/blob/master), and that should be installed according to [MATLAB Engine for Python Installation Instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Other options would be to modify 'get_syllable_duration_statistics' function to work without this or simply load precalculated values.

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

The project aims to use an XTTS model to synthesize Infant-Directed Speech (IDS) from IDS references, even though the model was not specifically designed for this purpose. Due to the poor quality of the IDS references, the model from [Resemble AI](https://github.com/resemble-ai/resemble-enhance) was employed to clean up the audio using the "denoise" and "enhance" functionalities.

Subsequently, prosodic and quality features were extracted from both the synthesized and reference audio. The extracted statistics were then visualized using various types of figures, including scatter plots with ellipse fitting to mean and standard deviation, Kernel Density Estimates (KDEs) from density functions, and radar plots summarizing all statistics together. Comparisons were made between the original IDS reference and the syntheses of all original, denoised, and enhanced versions.

## Examples

[Resemble AI](https://github.com/resemble-ai/resemble-enhance) for denoise and enhanced. [Example audio source](https://www.youtube.com/watch?v=F9qS_AZCdgM)

## References:

- **Original IDS:**

https://github.com/user-attachments/assets/9bdf8f6a-3642-4512-ae9b-fb93295dc2d4

- **Enhanced IDS:**

https://github.com/user-attachments/assets/dc57c506-9eb1-487f-b394-5ddfadc3f90e

- **Denoised IDS:**

https://github.com/user-attachments/assets/5ae9e06c-49e3-4232-b282-4a01952185ff

## Syntheses:

- **Original IDS:**

https://github.com/user-attachments/assets/531200fb-dfd2-4d78-a4a3-bd8d302c1cb0

- **Enhanced IDS:**

https://github.com/user-attachments/assets/0afd9964-d48c-4781-b17d-473e0751a1fb

- **Denoised IDS:**
  
https://github.com/user-attachments/assets/7814695c-132c-47d1-bb8f-413a7d08e3f8

### Figures

- **Scatter Plot:**
  ![Scatter Plot](examples/f0_sd_mean_scatter-1.png)

- **KDE of Density:**
  ![KDE of Density](examples/f0_kde_plot-1.png)

- **Logarithmic KDE:**
  ![Logarithmic KDE](examples/f0_logarithmic_kde_plot-1.png)

- **Radar Plot:**
  ![Radar Plot](examples/denoised_concat_9.wav_radar_plot-1.png)
