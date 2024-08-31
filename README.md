# IDS-Synthesis-with-XTTS
Analysing the IDS (Infant directed speech) cloning capabilities of a TTS model

## Installation:

1. Clone repository and install: https://github.com/orasanen/thetaOscillator/blob/master/thetaseg.m into "thetaOscillator" folder and follow the instructions there for installation.
   The matlab version was used here.

To get the same data the figures located in "figures" folder was used to create:

2. From .zip file (provided externally) extract the content of references into "references", synthesised into "synthesised" and texts into "texts" folders under "synthesis_stage".
3. Run statistics.py
4. Run display.py

To synthesize the data which was used to create the figures in "figures" folder:

2. Extract From .zip file (provided externally) extract the content of references into "references" and texts into "texts" folder under "synthesis_stage".
3. Install https://github.com/coqui-ai/TTS by following their insructions for installation. "XTTS" folder can be used to store the model(s).
4. Run synthesizer.py

To synthesize data from own samples (untested might not work):

2. Extract your references into the "original" folder under "synthesis_stage/references" and texts into the "texts" folder under "synthesis_stage"
3. Install https://github.com/coqui-ai/TTS by following their insructions for installation. "XTTS" folder can be used to store the model(s).
4. Run synthesizer.py
5. ????
6. Maybe it works.

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



