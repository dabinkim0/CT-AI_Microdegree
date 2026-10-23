"""Stable Audio 3 노트북(06, 07) 도우미.

CTxAI Stable Audio 3 MLX 패키지의 ctxai_audio.py를 Colab용으로 옮긴 것이다.
추론은 공식 저장소의 CPU 런타임(optimized/tflite, LiteRT)을 별도 프로세스로 실행한다.
GPU와 Hugging Face 로그인이 필요 없고, 가중치는 HF_HOME(= models/huggingface)에 받는다.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import ctai

SA3_URL = 'https://github.com/Stability-AI/stable-audio-3'
SA3_COMMIT = '77e0398c26f7d22d801ec1bb036307e8b231f741'   # MLX 패키지와 같은 커밋
SR = 44100
MAX_SECONDS = {'medium': 380, 'sm-music': 120}
CFG = dict(src=None, dit='medium', decoder='same-l', threads=2, out_root=None)
_SHOW = ('step ', '[1/3]', '[2/3]', '[3/3]', 'downloading', 'done ', 'error', 'Error', 'Traceback')


HF_REPO = 'stabilityai/stable-audio-3-optimized'
HF_REVISION = '08b17423649b5714b22eab20f96bf00751f2358a'   # MLX 패키지와 같은 가중치 revision
DIT_FILE = {'medium': 'tflite/sa3-m/dit_w8a8.tflite', 'sm-music': 'tflite/sa3-sm-music/dit_w8a8.tflite'}


def _ensure_weights(encoder=False):
    """필요한 가중치를 고정 revision으로 받아 공식 런타임이 찾는 위치에 연결한다.

    공식 런타임도 없는 파일을 스스로 받지만 revision을 고정하지 않으므로 여기서 먼저 받는다.
    """
    from huggingface_hub import hf_hub_download
    files = [DIT_FILE[CFG['dit']], 'tflite/t5gemma/encoder_fp16.tflite', f"tflite/{CFG['decoder']}/dec_w8a8.tflite"]
    if encoder:   # 인페인팅·변형에서만 필요
        files.append(f"tflite/{CFG['decoder']}/enc_w8a8.tflite")
    for name in files:
        target = CFG['src'] / 'models' / name
        if target.exists():
            continue
        print('내려받는 중:', name)
        cached = Path(hf_hub_download(HF_REPO, name, revision=HF_REVISION))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(cached)
        except OSError:
            import shutil
            shutil.copy2(cached, target)


def prepare(models_dir, out_dir, dit='medium', decoder='same-l', threads=None):
    """공식 코드를 받고, 가중치를 내려받고, 실행 설정을 정한다.

    코드와 CPU 캐시(.xnnwc)는 링크를 만들어야 해서 Drive 모드에서는 /content/code에 둔다.
    가중치 자체는 HF_HOME(= models/huggingface)에 있다.
    """
    code_root = Path('/content/code') if str(models_dir).startswith('/content/drive') else Path(models_dir)
    src = ctai.git_checkout(SA3_URL, code_root / 'stable-audio-3', SA3_COMMIT,
                            sparse=['optimized/tflite'])
    CFG.update(src=src / 'optimized' / 'tflite', dit=dit, decoder=decoder,
               threads=threads or (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) or 2,
               out_root=Path(out_dir) / 'stable_audio_3')
    _ensure_weights()
    import importlib.metadata as md
    print('LiteRT %s · 모델 %s + %s · CPU 스레드 %d' % (md.version('ai-edge-litert'), dit, decoder, CFG['threads']))


def audio_stats(path):
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if not np.isfinite(audio).all() or not len(audio):
        raise RuntimeError(f'Invalid audio: {path}')
    return {'sample_rate': sr, 'channels': audio.shape[1], 'frames': len(audio),
            'seconds': len(audio) / sr, 'peak': float(np.abs(audio).max()),
            'rms': float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))}


def show_audio(path):
    from IPython.display import Audio, display
    display(Audio(filename=str(path)))
    return audio_stats(path)


def plot_audio(path, region=None):
    import matplotlib.pyplot as plt
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    stride = max(1, len(audio) // 15000)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.arange(0, len(audio), stride) / sr, audio[::stride, 0], lw=0.5, label='Left channel')
    if region is not None:
        ax.axvspan(*region, color='orange', alpha=0.3, label='Edit region')
    ax.set(xlabel='Time (s)', ylabel='Amplitude', title=Path(path).name)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_spectrogram(audio, sr, n_fft=2048, hop=512, n_mels=128):
    """좌우 평균 STFT 파워에 HTK 삼각 멜 필터 128개 → dB (기준 1, 클립별 정규화 없음)."""
    from scipy.signal import stft
    if audio.ndim != 2 or len(audio) < n_fft:
        raise ValueError('Expected at least 2048 frames of channel-last audio.')
    frequencies, times, spectrum = stft(
        audio.T, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop,
        boundary=None, padded=False, scaling='spectrum', axis=-1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    edges = 700.0 * (10 ** (np.linspace(0, _hz_to_mel(sr / 2), n_mels + 2) / 2595.0) - 1)
    rising = (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None]
    filters = np.maximum(0.0, np.minimum(rising, falling))
    mel_power = filters @ power
    return edges[1:-1], times, 10 * np.log10(np.maximum(mel_power, 1e-12))


def compare_audio(clips, *, region=None, crop=None, title='Audio comparison',
                  output_dir=None, filename='comparison', vmin=-90, vmax=-10):
    """같은 길이의 클립들을 같은 멜 축·색 범위로 비교하고 PNG/HTML/JSON으로 저장한다.

    clips: [(label, WAV 경로), ...]. region은 편집 구간 표시, crop은 그 구간만 그리고 재생.
    """
    import base64
    import html
    import io
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    if not 2 <= len(clips) <= 8:
        raise ValueError('Compare 2–8 clips at a time.')
    records, specs, cards = [], [], []
    expected = None
    for label, path in clips:
        path = Path(path).resolve()
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        shape = (sr, *audio.shape)
        if expected is not None and shape != expected:
            raise ValueError('For fair comparison all inputs must have equal sample rate, length and channels.')
        expected = shape
        duration = len(audio) / sr
        for bounds in (crop, region):
            if bounds is not None and not 0 <= bounds[0] < bounds[1] <= duration:
                raise ValueError('Time range must lie within the audio.')
        left, right = crop if crop is not None else (0, duration)
        segment = audio[round(left * sr):round(right * sr)]
        f, t, db = _mel_spectrogram(segment, sr)
        specs.append((str(label), f, t + left, db))
        buffer = io.BytesIO()
        sf.write(buffer, segment, sr, format='WAV', subtype='PCM_16')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        peak = float(np.abs(segment).max())
        records.append({'label': str(label), 'path': str(path), 'display_range': [left, right],
                        'rms': rms, 'peak': peak,
                        'clipped_fraction': float(np.mean(np.abs(segment) >= 32767 / 32768))})
        cards.append(f'<section style="padding:12px;border:1px solid #888;border-radius:8px">'
                     f'<strong>{html.escape(str(label))}</strong>'
                     f'<p>{left:g}–{right:g} s · RMS {rms:.4f} · Peak {peak:.4f}</p>'
                     f'<audio controls preload="none" style="width:100%" '
                     f'src="data:audio/wav;base64,{encoded}"></audio></section>')

    columns = 2
    rows = math.ceil(len(specs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.7 * rows),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    for ax, (label, f, t, db) in zip(axes.flat, specs):
        mesh = ax.pcolormesh(t, _hz_to_mel(f), db, shading='auto',
                             cmap='magma', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set(title=label, xlabel='Time (s)', ylabel='Frequency (Hz, mel scale)',
               xlim=(left, right), ylim=(0, _hz_to_mel(sr / 2)))
        ticks = np.array([0, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_yticks(_hz_to_mel(ticks), [f'{hz:g}' for hz in ticks])
        if region is not None:
            for boundary in region:
                if left <= boundary <= right:
                    ax.axvline(boundary, color='#45e8ff', ls='--', lw=1.4)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    fig.suptitle(f'{title} | Mel spectrogram', fontsize=14)
    fig.colorbar(mesh, ax=[ax for ax in axes.flat if ax.get_visible()],
                 label='Mel power (dB re 1; shared scale)', shrink=0.9)
    directory = Path(output_dir) if output_dir else _job('compare')
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f'{filename}.png'
    fig.savefig(png, dpi=150)
    plt.show()
    plt.close(fig)
    player_html = ('<div style="display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:12px">'
                   + ''.join(cards) + '</div>')
    display(HTML(player_html))
    graph = base64.b64encode(png.read_bytes()).decode('ascii')
    (directory / f'{filename}.html').write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>{html.escape(title)}</title><body style="font-family:system-ui;margin:24px">'
        f'<h1>{html.escape(title)}</h1><img style="max-width:100%" src="data:image/png;base64,{graph}">'
        + player_html + '</body>')
    report = {'title': title, 'clips': records, 'region': region, 'crop': crop,
              'spectrogram': {'type': 'mel', 'n_mels': 128, 'n_fft': 2048, 'hop': 512,
                              'vmin_db': vmin, 'vmax_db': vmax},
              'audio_normalization': False, 'png': str(png)}
    (directory / f'{filename}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _job(mode):
    if CFG['src'] is None:
        raise RuntimeError('먼저 sa3.prepare(...)를 실행하세요.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = CFG['out_root'] / f'{stamp}_{mode}_{uuid.uuid4().hex[:6]}'
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _validate(prompt, seconds, seed, steps, cfg, negative_prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('A non-empty prompt is required.')
    limit = MAX_SECONDS.get(CFG['dit'], 120)
    if not math.isfinite(seconds) or not 1 <= seconds <= limit:
        raise ValueError(f'Duration must be 1–{limit} seconds.')
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('seed must be an integer in [0, 2**31).')
    if not isinstance(steps, int) or not 1 <= steps <= 32:
        raise ValueError('steps must be 1–32; the distilled model defaults to 8.')
    if not math.isfinite(cfg) or not 0 <= cfg <= 10:
        raise ValueError('cfg must be between 0 and 10.')
    if negative_prompt and cfg == 1:
        raise ValueError('negative_prompt is ignored at cfg=1. Set cfg>1 or clear negative_prompt.')


def _infer(directory, *, prompt, seconds, seed, steps, cfg, negative_prompt='',
           init_audio=None, edit_range=None, noise_level=1.0, timeout=3600):
    _validate(prompt, seconds, seed, steps, cfg, negative_prompt)
    if not 0.01 <= noise_level <= 1:
        raise ValueError('noise_level must be between 0.01 and 1.')
    out = directory / 'model_output.wav'
    command = [sys.executable, '-u', str(CFG['src'] / 'scripts' / 'sa3_tflite.py'),
               '--dit', CFG['dit'], '--decoder', CFG['decoder'],
               '--prompt', prompt, '--seconds', str(seconds), '--seed', str(seed),
               '--steps', str(steps), '--cfg', str(cfg), '--threads', str(CFG['threads']),
               '--out', str(out)]
    if negative_prompt:
        command += ['--negative-prompt', negative_prompt]
    if init_audio:
        command += ['--init-audio', str(init_audio), '--init-noise-level', str(noise_level)]
    if edit_range:
        command += ['--inpaint-range', f'{edit_range[0]},{edit_range[1]}']
    meta = dict(prompt=prompt, seconds=seconds, seed=seed, steps=steps, cfg=cfg,
                negative_prompt=negative_prompt, init_audio=str(init_audio) if init_audio else None,
                edit_range=edit_range, noise_level=noise_level, model=CFG['dit'], decoder=CFG['decoder'],
                source_commit=SA3_COMMIT, command=command, status='running')
    manifest = directory / 'run.json'
    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    start = time.monotonic()
    process = None
    try:
        with (directory / 'inference.log').open('w') as log:
            process = subprocess.Popen(command, cwd=CFG['src'], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            with (directory / 'inference.log').open(errors='replace') as reader:
                buf = ''
                while True:
                    finished = process.poll() is not None
                    buf += reader.read()
                    *lines, buf = buf.replace('\r', '\n').split('\n')
                    for line in lines:   # LiteRT의 INFO 로그는 숨기고 진행 상황만 보여줌
                        if any(k in line for k in _SHOW) and not line.lstrip().startswith(('INFO:', 'WARNING:')):
                            print(line.strip()[:150], flush=True)
                    if finished:
                        break
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f'Inference exceeded {timeout}s; see {directory}')
                    time.sleep(0.5)
        if process.returncode:
            raise RuntimeError(f'Inference exited {process.returncode}; see {directory / "inference.log"}')
        stats = audio_stats(out)
        if stats['sample_rate'] != SR or stats['channels'] != 2:
            raise RuntimeError(f'Unexpected output audio format: {stats}')
        meta.update(status='complete', stats=stats)
        return out
    except BaseException as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        meta['wall_seconds'] = round(time.monotonic() - start, 2)
        manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def generate(prompt, *, seconds=20, seed=42, steps=8, cfg=1.0, negative_prompt=''):
    return _infer(_job('generate'), prompt=prompt, seconds=seconds, seed=seed,
                  steps=steps, cfg=cfg, negative_prompt=negative_prompt)


def generate_cfg_sweep(prompt, *, cfg_values=(1, 4), seconds=20, seed=42, steps=8, negative_prompt=''):
    """프롬프트·시드·길이·스텝을 고정하고 CFG만 바꿔 차례로 생성한다."""
    values = list(cfg_values)
    if not 2 <= len(values) <= 8 or len(set(values)) != len(values):
        raise ValueError('Provide 2–8 distinct CFG values.')
    for cfg in values:
        _validate(prompt, seconds, seed, steps, cfg, negative_prompt)
    directory = _job('cfg_sweep')
    manifest = {'prompt': prompt, 'seed': seed, 'seconds': seconds, 'steps': steps,
                'negative_prompt': negative_prompt, 'cfg_values': values, 'status': 'running', 'runs': []}
    save = lambda: (directory / 'sweep.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    save()
    clips = []
    try:
        for index, cfg in enumerate(values):
            run_dir = directory / f'cfg_{index:02d}_{cfg:g}'
            run_dir.mkdir()
            print(f'\n=== CFG {cfg:g} | seed {seed} ===', flush=True)
            clip = _infer(run_dir, prompt=prompt, seconds=seconds, seed=seed, steps=steps,
                          cfg=cfg, negative_prompt=negative_prompt)
            clips.append((f'CFG = {cfg:g}', clip))
            manifest['runs'].append({'cfg': cfg, 'audio': str(clip)})
            save()
        manifest['status'] = 'complete'
        return directory, clips
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        save()


def prepare_audio(source, destination, max_seconds=None):
    """입력을 44.1 kHz 스테레오 PCM16 기준 파일로 바꾼다. 원본은 건드리지 않는다."""
    import librosa
    source, destination = Path(source).expanduser().resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    y, _ = librosa.load(str(source), sr=SR, mono=False, duration=max_seconds)
    y = np.stack([y, y]) if y.ndim == 1 else y[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), y.T, SR, subtype='PCM_16')
    seconds = sf.info(str(destination)).duration
    limit = MAX_SECONDS.get(CFG['dit'], 120)
    if not 1 <= seconds <= limit:
        raise ValueError(f'Input must be 1–{limit}s.')
    return destination


def composite_region(reference, generated, output, start, end, crossfade=0.1):
    """[start, end) 밖의 PCM16 샘플은 그대로 두고, 안쪽만 모델 출력으로 바꾼다(경계 안쪽 크로스페이드)."""
    src, sr = sf.read(str(reference), dtype='int16', always_2d=True)
    new, new_sr = sf.read(str(generated), dtype='int16', always_2d=True)
    if new_sr != sr:
        raise ValueError('Sample rates differ.')
    if len(new) < len(src):   # 모델 출력이 짧으면 0으로 채움
        new = np.concatenate([new, np.zeros((len(src) - len(new), new.shape[1]), new.dtype)])
    new = new[:len(src)]
    a, b = round(start * sr), round(end * sr)
    result = src.copy()
    n = min(round(crossfade * sr), (b - a) // 2)
    alpha = np.ones((b - a, 1))
    if n:
        alpha[:n, 0] = np.linspace(0, 1, n)
        alpha[-n:, 0] = np.linspace(1, 0, n)
    result[a:b] = np.rint(src[a:b] * (1 - alpha) + new[a:b] * alpha).clip(-32768, 32767).astype(np.int16)
    sf.write(str(output), result, sr, subtype='PCM_16')
    saved, _ = sf.read(str(output), dtype='int16', always_2d=True)
    preserved = np.array_equal(src[:a], saved[:a]) and np.array_equal(src[b:], saved[b:])
    if not preserved:
        raise RuntimeError('Unexpected change outside the selected range.')
    return {'outside_region_bit_exact': preserved, 'start_sample': a, 'end_sample': b,
            'crossfade_samples': n, 'edited_region_changed': not np.array_equal(src[a:b], saved[a:b])}


def inpaint(source, prompt, *, start, end, seed=43, steps=8, cfg=1.0,
            negative_prompt='', crossfade=0.1, max_seconds=None):
    _ensure_weights(encoder=True)
    if not 0 <= start < end:
        raise ValueError('Require 0 <= start < end.')
    if end - start < 0.2:
        raise ValueError('Select at least 0.2 seconds.')
    directory = _job('inpaint')
    prepared = prepare_audio(source, directory / 'reference_44100.wav', max_seconds)
    seconds = sf.info(str(prepared)).duration
    if end > seconds:
        raise ValueError(f'Edit range ends at {end}s but input is {seconds:.3f}s.')
    raw = _infer(directory, prompt=prompt, seconds=round(seconds, 3), seed=seed, steps=steps,
                 cfg=cfg, negative_prompt=negative_prompt, init_audio=prepared,
                 edit_range=(start, end))
    output = directory / 'edited.wav'
    report = composite_region(prepared, raw, output, start, end, crossfade)
    report.update(original_input=str(Path(source).resolve()), reference=str(prepared),
                  raw_model_output=str(raw), output=str(output))
    (directory / 'edit_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ('outside_region_bit_exact', 'edited_region_changed',
                                              'start_sample', 'end_sample')}, ensure_ascii=False))
    return output


def variation(source, prompt, *, noise_level=0.6, seed=44, steps=8, cfg=1.0, max_seconds=None):
    _ensure_weights(encoder=True)
    directory = _job('variation')
    prepared = prepare_audio(source, directory / 'reference_44100.wav', max_seconds)
    return _infer(directory, prompt=prompt, seconds=round(sf.info(str(prepared)).duration, 3),
                  seed=seed, steps=steps, cfg=cfg, init_audio=prepared, noise_level=noise_level)
