# Regression source

`speech.wav` is the original synthetic recording generated locally on 2026-10-09 using Windows `System.Speech.Synthesis.SpeechSynthesizer` and its default voice. Speech-engine rights remain with its vendor; this is not a downloaded human-speech dataset.

Reference: "Hello everyone. This is a local audio transcription demonstration. The system preserves the timing of the original recording."

Integration tests derive resampled, quiet, pause, silence, noise, and tone fixtures in pytest temporary storage. Demo recordings live separately in `sample_audio/`.
