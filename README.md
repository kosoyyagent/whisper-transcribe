# Whisper Transcribe Tools

Two small CLI tools for freelance/team transcription work, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper):

## `transcribe.py` — verbatim transcript with timestamps

```bash
python transcribe.py input.mp4 -o out.txt            # [HH:MM:SS] per phrase
python transcribe.py input.mp4 --srt subs.srt        # also writes SRT subtitles
python transcribe.py input.mp3 --model large-v3      # best accuracy (default)
python transcribe.py input.mp3 --device cpu --compute int8   # no GPU needed
python transcribe.py input.mp3 --verbatim            # keep «эээ», stutters
```

Output format:

```
[00:00:00] Первая фраза диктора.
[00:00:07] Вторая фраза.
```

## `dialogue_sheet.py` — dialogue montage sheet with speaker diarization

```bash
python dialogue_sheet.py interview.mp4 -o sheet.txt
python dialogue_sheet.py interview.mp3 --speakers 2 --swap
```

Output: `№ | timecode | Speaker | line` — speakers are auto-separated by
MFCC clustering (`--speakers N`); "Интервьюер/Гость" roles guessed by
speech share and questions (`--swap` to flip). See
`docs/sample_output.txt` for a real example.

## Requirements

```bash
pip install -r requirements.txt
```

- `transcribe.py` needs only `faster-whisper`.
- `dialogue_sheet.py` additionally needs `av`, `librosa`, `scikit-learn`,
  `numpy` — audio is decoded via PyAV, no external ffmpeg binary required.
- GPU (`--device cuda --compute float16`) for speed, or
  `--device cpu --compute int8` on a laptop — a `tiny`/`base` model still
  transcribes short clips in seconds.

## Notes

- Default language is Russian (`--lang ru`), any whisper-supported code
  works.
- First run downloads the selected model from HuggingFace (`tiny` ≈75 MB,
  `large-v3` ≈3 GB).
- Speaker labels are heuristics — check the final sheet before handing
  it to a client (that's the honest workflow this tool is built for).

## License

MIT — see LICENSE.
