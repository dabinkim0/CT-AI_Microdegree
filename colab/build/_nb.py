"""Tiny helper to build notebooks from (kind, source) lists with nbformat."""
import nbformat as nbf
import os
NB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks')
def build(path, cells, title, gpu=True):
    path = os.path.join(NB_DIR, os.path.basename(path))
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"name": title, "provenance": [], "toc_visible": True}}
    if gpu: nb.metadata["accelerator"] = "GPU"
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path); print("wrote", path, len(cells), "cells")

BOOTSTRAP = r'''#@title 0. 준비: 수업 자료 받기와 모델 폴더 정하기 (실행만 하면 됩니다)
#@markdown 모델 가중치는 모두 `models` 폴더 하나에 받습니다. 기본 위치는 `/content`이고 런타임을 끄면 지워집니다. 체크하면 Google Drive의 `CT-AI_Microdegree/` 아래에 `models`와 `outputs`를 두어 다음 실행 때 다시 받지 않고, 다른 노트북의 결과도 이어서 쓸 수 있습니다.
KEEP_IN_DRIVE = False  #@param {type:"boolean"}
import os, sys, glob, time, json, subprocess
_cands = [os.environ.get('CTAI_REPO_DIR'), os.path.abspath(os.path.join(os.getcwd(), '..', '..')), '/content/CT-AI_Microdegree']
REPO = next((c for c in _cands if c and os.path.isfile(os.path.join(c, 'colab', 'ctai.py'))), None)
if REPO is None:
    REPO = '/content/CT-AI_Microdegree'
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/dabinkim0/CT-AI_Microdegree', REPO], check=True)
sys.path.insert(0, os.path.join(REPO, 'colab'))
REQ = os.path.join(REPO, 'colab', 'requirements')   # 노트북별 pip 패키지 목록
import ctai
P = ctai.setup(keep_in_drive=KEEP_IN_DRIVE)
AUDIO_DIR, ASSET_DIR, MODELS_DIR, CODE_DIR, OUT_DIR, IN_COLAB = P['AUDIO_DIR'], P['ASSET_DIR'], P['MODELS_DIR'], P['CODE_DIR'], P['OUT_DIR'], P['IN_COLAB']
print('예시 음원:', sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, '*.wav'))))
'''

AUDIO_PICK = r'''#@title 오디오 준비: 예시 음원 / 업로드 / YouTube 중 선택
import ipywidgets as W, soundfile as sf, numpy as np, librosa
from IPython.display import Audio, display
import matplotlib.pyplot as plt

EXAMPLES = {EXAMPLES}
SOURCE = '예시'      #@param ["예시", "업로드", "YouTube"]
EXAMPLE = {DEFAULT_EXAMPLE!r}   #@param {EXAMPLE_PARAM}
YOUTUBE_URL = ''   #@param {{type:"string"}}
MAX_SECONDS = {MAX_SECONDS}   #@param {{type:"integer"}}
TARGET_SR = {TARGET_SR}

def load_audio(path, sr=TARGET_SR, max_sec=MAX_SECONDS, mono={MONO}):
    y, _ = librosa.load(path, sr=sr, mono=mono)
    if max_sec: y = y[..., :int(max_sec * sr)]
    return y, sr

if SOURCE == '예시':
    audio_path = os.path.join(AUDIO_DIR, EXAMPLES[EXAMPLE])
elif SOURCE == '업로드':
    audio_path = str(ctai.upload())   # Colab: 파일 선택 창 / 로컬: 파일 경로 입력
else:
    # 수업 내 학습 목적으로만 사용하세요. 저작권이 있는 음원은 외부에 배포하지 않습니다.
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'yt-dlp'], check=True)
    subprocess.run(['yt-dlp', '-q', '-x', '--audio-format', 'wav', '-o', 'yt_input.%(ext)s', YOUTUBE_URL], check=True)
    audio_path = 'yt_input.wav'

y, sr = load_audio(audio_path)
mono = y if y.ndim == 1 else y.mean(0)
print('파일:', audio_path, '| 길이: %.1f s | sr=%d' % (len(mono) / sr, sr))
plt.figure(figsize=(14, 2)); plt.plot(np.arange(len(mono)) / sr, mono, lw=.4, color='#58c4dd'); plt.xlabel('time (s)'); plt.title('waveform'); plt.tight_layout(); plt.show()
display(Audio(mono, rate=sr))
'''

SINE_HELPERS = r'''#@title 합성 도우미: note 목록을 사인파로 바꿔 듣기 (데모 페이지의 합성과 같은 방식)
import numpy as np
def midi_to_hz(m): return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)
def synth_notes(notes, sr=22050, duration=None, gain=0.12, drums=False):
    """notes: [(onset, offset, pitch), ...] → float32 waveform. drums=True면 노이즈 버스트."""
    if duration is None: duration = max(n[1] for n in notes) + 0.5 if notes else 1.0
    out = np.zeros(int(duration * sr) + 1, dtype=np.float32)
    rng = np.random.default_rng(0)
    for on, off, pitch in notes:
        a = int(on * sr); b = int(max(off, on + 0.08) * sr) + int(0.05 * sr)
        if a >= len(out): continue
        n = min(b, len(out)) - a; t = np.arange(n) / sr
        if drums:
            kick = pitch in (35, 36); dec = 0.18 if kick else 0.08
            env = np.exp(-t / dec); sig = rng.standard_normal(n) * 0.5
            if kick: sig = sig * 0.3 + np.sin(2 * np.pi * (150 * np.exp(-t / 0.1) + 45) * t) * 0.9
            out[a:a + n] += (sig * env * gain * 3).astype(np.float32)
        else:
            env = np.minimum(1, t / 0.008) * np.where(t < 0.25, 1 - 0.65 * t / 0.25, 0.35)
            env[-int(0.03 * sr):] *= np.linspace(1, 0, int(0.03 * sr))
            out[a:a + n] += (np.sin(2 * np.pi * midi_to_hz(pitch) * t) * env * gain).astype(np.float32)
    peak = np.abs(out).max(); return out / peak * 0.8 if peak > 0 else out
def mix(*tracks):
    n = max(len(t) for t in tracks); s = np.zeros(n, np.float32)
    for t in tracks: s[:len(t)] += t
    peak = np.abs(s).max(); return s / peak * 0.8 if peak > 0 else s
'''
