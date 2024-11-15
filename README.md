# IDS-Synthesis-with-XTTS
Analyzing the IDS (Infant-Directed Speech) synthesizing capabilities of a TTS model

## Installation:
### Start here

Clone the repository and install [thetaOscillator](https://github.com/orasanen/thetaOscillator/blob/master) into the "thetaOscillator" folder and follow the instructions provided there for installation. The MATLAB version was used here.

### Extract data from existing set of synthesis and their references ->

1. From the provided .zip file, extract the contents into STAGE/synthesis. References go into references and syntheses go to output. After, follow the pipeline_readme.txt in SCRIPTS/python/main/extraction-pipeline.

### Synthesize and extract statistics->

1. From the provided .zip file, extract the contents of the references into the "references" folder and texts into the "texts" folder under "synthesis_stage".
2. Install [Coqui TTS](https://github.com/coqui-ai/TTS) by following their installation instructions. The "XTTS" folder can be used to store the model(s).
3. Run `synthesizer.py`
4. Follow the pipeline_readme.txt in SCRIPTS/python/main/extraction-pipeline.

## Project Description

The project aims to use an XTTS model to synthesize Infant-Directed Speech (IDS) from IDS references, even though the model was not specifically designed for this purpose. Due to the poor quality of the IDS references, the model from [Resemble AI](https://github.com/resemble-ai/resemble-enhance) was employed to clean up the audio using the "denoise" and "enhance" functionalities.

Subsequently, prosodic and quality features were extracted from both the synthesized and reference audio. The extracted statistics were then visualized using various types of figures, including scatter plots with ellipse fitting to mean and standard deviation, Kernel Density Estimates (KDEs) from density functions, and radar plots summarizing all statistics together. Comparisons were made between the original IDS reference and the syntheses of all original, denoised, and enhanced versions.
