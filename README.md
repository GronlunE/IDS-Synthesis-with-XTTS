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

Project intention is to attempt to use a XTTS model to synthesize IDS from IDS references, even if it was not specifically designed to do be able to do so. 
The IDS references used were of poor quality so model from https://github.com/resemble-ai/resemble-enhance was used to clean up audio, with their "denoise" and "enhance" functionalities
Then the synthesized and reference audio had chosen relevant prosodic and quality features extracted. Figures were then drawn from the extracted statistics. Figure type currently used are scatter + ellipse fitting to mean and standart deviation, kdes from density functions and radar plots of all statistics together. Comparison base was using the Original IDS reference against the syntheses of all orginal, denoised and enhanced.

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



