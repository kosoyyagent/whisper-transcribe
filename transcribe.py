#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""transcribe.py — дословная расшифровка аудио/видео в текст с тайм-кодами.

Запуск:
  python transcribe.py input.mp4 -o out.txt            # фразы с тайм-кодом 00:00:00
  python transcribe.py input.mp4 -o out.txt --verbatim  # «эээ»/повторы оставить
  python transcribe.py input.mp4 --model large-v3      # точнее, медленнее
  python transcribe.py input.mp4 --srt out.srt         # плюс файл субтитров

Формат вывода:
  [00:00:00] текст фразы
  [00:00:07] следующая фраза
"""
import argparse
import os
import sys


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(t, verbatim):
    t = t.strip()
    if verbatim:
        return t
    # лёгкая чистка обрывов-заполнителей, если модель их выделила отдельными токенами
    for w in ("эээ", "э-э", "м-м", "а-а", "хм"):
        if t.lower().rstrip('.…, ') == w:
            return ''
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--lang', default='ru')
    ap.add_argument('--verbatim', action='store_true', help='оставлять все заполнители дословно')
    ap.add_argument('--srt', default=None, help='также записать SRT-файл')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--compute', default='float16')
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    print(f"[load] {args.model} on {args.device}", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    segments, info = model.transcribe(
        args.input,
        language=args.lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    print(f"[info] lang={info.language} p={info.language_probability:.2f} dur={info.duration:.0f}s", file=sys.stderr)

    lines, srt = [], []
    n = 0
    for seg in segments:
        txt = clean_text(seg.text, args.verbatim)
        if not txt:
            continue
        n += 1
        lines.append(f"[{fmt_ts(seg.start)}] {txt}")
        srt.append(f"{n}\n{fmt_srt_ts(seg.start)} --> {fmt_srt_ts(seg.end)}\n{txt}\n")

    out = args.out or os.path.splitext(args.input)[0] + '.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[done] {n} phrases -> {out}", file=sys.stderr)
    if args.srt:
        with open(args.srt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt))
        print(f"[done] srt -> {args.srt}", file=sys.stderr)


if __name__ == '__main__':
    main()
