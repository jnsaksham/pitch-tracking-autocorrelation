# Pitch tracking using autocorrelation

A pitch-tracking system that implements autocorrelation function in the time domain.

The system first detects silence in audio above a custom dB threshold, and applies the mask.
It then estimates pitch for every frame in the audio.

A few tests are also provided to evaluate the system in the same file.
