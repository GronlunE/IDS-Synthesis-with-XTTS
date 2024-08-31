# IDS-Synthesis-with-XTTS
Analysing the IDS (Infant directed speech) cloning capabilities of a TTS model

## Installation:
### Start here 

1. Clone repository and install: https://github.com/orasanen/thetaOscillator/blob/master/thetaseg.m into "thetaOscillator" folder and follow the instructions there for installation.
   The matlab version was used here. The enviroment variable does not contain TTS.
2. ->

### ONLY EXTRACT DATA ->


2. From the .zip file (provided externally) extract the content of references into "references", synthesised into "synthesised" and texts into "texts" folders under "synthesis_stage".
3. Run statistics.py
4. Run display.py


### Synthesize with the SAME references as I ->


2. From the .zip file (provided externally) extract the content of references into "references" and texts into "texts" folder under "synthesis_stage".
3. Install https://github.com/coqui-ai/TTS by following their insructions for installation. "XTTS" folder can be used to store the model(s).
4. Run synthesizer.py
5. Run statistics.py
6. Run display.py


### Synthesize with DIFFERENT references as I ->


2. Extract your references into the "original" folder under "synthesis_stage/references" and texts into the "texts" folder under "synthesis_stage"
3. Install https://github.com/coqui-ai/TTS by following their insructions for installation. "XTTS" folder can be used to store the model(s).
4. Run synthesizer.py
5. ????


## Project description

The project aims to use an XTTS model to synthesize Infant-Directed Speech (IDS) from IDS references, even though the model was not specifically designed for this purpose. Due to the poor quality of the IDS references, the model from Resemble AI was employed to clean up the audio using their "denoise" and "enhance" functionalities.

Subsequently, prosodic and quality features were extracted from both the synthesized and reference audio. The extracted statistics were then visualized using various types of figures, including scatter plots with ellipse fitting to mean and standard deviation, Kernel Density Estimates (KDEs) from density functions, and radar plots summarizing all statistics together. Comparisons were made between the original IDS reference and the syntheses of all original, denoised, and enhanced versions.



## Examples

### References:

Orignal IDS:
Enhanced IDS:
Denoised IDS:

### Syntheses:

Orignal IDS:
Enhanced IDS:
Denoised IDS:

### Figures

Scatter:
KDE of Density:
Radar:



