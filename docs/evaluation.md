# Evaluation Logs and Outputs

## What was evaluated
- MIDI loading and note extraction
- Sequence window generation for training
- Transformer-style next-note prediction
- MIDI export path for generated sequences

## Observable outputs
- The repository already includes a generated artifact: `generated_music.mid`
- Training produces a note-prediction model and a generated note sequence

## Example console-style output
```text
Scanning dataset folder
Loading MIDI file: ...
Num GPUs Available:  1
MIDI file successfully saved as generated_music.mid
```

## Notes
- No quantitative music-quality evaluation or listener study is committed in the repo.
- The UI exposes raga, tala, and instrument selectors, but the current scripts do not use them as explicit conditioning inputs.
