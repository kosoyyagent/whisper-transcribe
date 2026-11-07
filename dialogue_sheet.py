#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dialogue_sheet.py — диалоговый монтажный лист из аудио/видео.

Формат вывода (txt):
  № | тайм-код | Спикер | Реплика

Спикеры определяются автоматически (MFCC-кластеризация, 2 голоса).
Роли «Интервьюер/Гость» угадываются по доле речи и вопросам — поправить вручную
перед сдачей не нужно, но при желании есть --swap.

Запуск:
  python dialogue_sheet.py input.mp4 -o sheet.txt
  python dialogue_sheet.py input.mp3 -o sheet.txt --speakers 2 --model medium
"""
import argparse
import os
import sys


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def decode_audio(path, sr=16000):
    """Decode to mono float32 via PyAV (no external ffmpeg needed)."""
    import av
    import numpy as np
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format='flt', layout='mono', rate=sr)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            arr = rf.to_ndarray()
            chunks.append(arr.reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype='float32')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--model', default='medium')
    ap.add_argument('--lang', default='ru')
    ap.add_argument('--speakers', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--compute', default='float16')
    ap.add_argument('--swap', action='store_true',
                    help='поменять Интервьюер/Гость местами')
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    print(f"[load] whisper {args.model}", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)
    segments, info = model.transcribe(
        args.input, language=args.lang, beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True, condition_on_previous_text=True)
    segs = [s for s in segments if s.text.strip()]
    print(f"[info] {len(segs)} segments, dur={info.duration:.0f}s", file=sys.stderr)

    # --- speaker diarization via MFCC clustering ---
    import numpy as np
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    print("[load] audio decode", file=sys.stderr)
    y = decode_audio(args.input)
    sr = 16000

    feats, kept = [], []
    for s in segs:
        a, b_ = int(s.start * sr), max(int(s.end * sr), int(s.start * sr) + 4000)
        frag = y[a:b_]
        if len(frag) < 2048:
            continue
        mfcc = librosa.feature.mfcc(y=frag, sr=sr, n_mfcc=20)
        v = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        feats.append(v)
        kept.append(s)

    labels = [0] * len(kept)
    if len(kept) >= args.speakers:
        X = StandardScaler().fit_transform(np.array(feats))
        labels = AgglomerativeClustering(n_clusters=args.speakers).fit_predict(X)
    for s, l in zip(kept, labels):
        s.speaker = int(l)

    # merge consecutive same-speaker turns
    turns = []
    for s in kept:
        if turns and turns[-1]['sp'] == s.speaker and s.start - turns[-1]['end'] < 2.0:
            turns[-1]['text'] += ' ' + s.text.strip()
            turns[-1]['end'] = s.end
        else:
            turns.append({'sp': s.speaker, 'start': s.start, 'end': s.end,
                          'text': s.text.strip()})

    # guess roles: Интервьюер = меньшая доля речи или больше '?' на реплику
    from collections import defaultdict
    talk, ques = defaultdict(float), defaultdict(int)
    for t in turns:
        talk[t['sp']] += t['end'] - t['start']
        ques[t['sp']] += t['text'].count('?')
    spk_ids = sorted(talk, key=talk.get)
    interviewer = spk_ids[0]  # меньше всех говорит = интервьюер
    if ques[interviewer] == 0 and ques and max(ques.values()) > 0:
        interviewer = max(ques, key=ques.get)
    if args.swap:
        interviewer = 1 - interviewer

    names = {interviewer: 'Интервьюер'}
    for sp in set(t['sp'] for t in turns):
        names.setdefault(sp, 'Гость' if len(set(t['sp'] for t in turns)) == 2 else f'Спикер {sp+1}')

    lines = ['ДИАЛОГОВЫЙ МОНТАЖНЫЙ ЛИСТ', f'Файл: {os.path.basename(args.input)}', '']
    for i, t in enumerate(turns, 1):
        lines.append(f"{i:>4} | {fmt_ts(t['start'])} | {names[t['sp']]} | {t['text']}")

    out = args.out or os.path.splitext(args.input)[0] + '_sheet.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[done] {len(turns)} turns -> {out}", file=sys.stderr)


if __name__ == '__main__':
    main()
