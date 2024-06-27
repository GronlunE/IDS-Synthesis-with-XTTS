# Voice-cloning-Analysis
Analysing the voice cloning capabilities of a TTS model

How to measure: 

1. Auditory perception (speech style, speaker identity, noisiness)
2. Praat analysis (intonation).

What we want to know:

1. Stability in general, e.g., does the speaker change if the text changes but the reference audio remains the same? Does the speaker or speech style change if the reference audio is shortened or otherwise manipulated?
2. Does the prosody of infant-directed speech (IDS) transfer to synthesis, and does the noise transfer as well? Does the prosody change if the reference IDS sample is replaced with another from the same speaker, etc?
3. Does the prosody transfer if the reference audio is cleaned, either by machine learning or filtering?

[Listen to the audio file](audio/xtts_empty_story.wav)

Or use the embedded player below:

<audio controls>
  <source src="audio/xtts_empty_story.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>
