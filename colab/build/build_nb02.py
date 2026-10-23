"""Builds colab/notebooks/02_YourMT3plus_multi_instrument_transcription.ipynb (outputs stripped)."""
import nbformat as nbf
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import BOOTSTRAP
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "02_YourMT3plus_multi_instrument_transcription.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------------------------------
md(r"""
# 02 · YourMT3+ — 음악 한 곡을 악기별 MIDI로 받아 적기

**YourMT3+** (Chang et al., 2024): 믹스된 음악 오디오를 듣고 **어떤 악기가 언제 어떤 음을 냈는지**를 음표 목록(MIDI)으로 받아 적는 멀티악기 자동 채보(AMT) 모델.

| | |
|---|---|
| 입력 → 출력 | 오디오(wav/mp3, 아무 샘플레이트) → 악기 라벨이 붙은 음표 목록 (onset, offset, pitch, program) → **멀티트랙 MIDI** |
| 모델 | `YPTF.MoE+Multi (noPS)` — Perceiver-TF 인코더 + MoE, 13채널 T5 디코더, 약 47 M 파라미터, 체크포인트 561 MB |
| 논문 / 코드 | [arXiv:2407.04822](https://arxiv.org/abs/2407.04822) · [HF Space mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3) |
| 예상 시간 (Colab T4) | 설치 ≈ 2분 · 체크포인트 다운로드 ≈ 1분 · 60 초 음악 추론 ≈ 20~40 초 |
| GPU | 권장 (런타임 → 런타임 유형 변경 → **T4 GPU**). CPU로도 동작하지만 60 초 음악에 약 2~3분 |

진행 순서: 준비 → 설치 → 음악 고르기 → 모델 불러오기 → 추론 → 피아노롤 보기·듣기 → 손잡이 돌려보기 → 저장
""")
# ----------------------------------------------------------------------------------------------
md(r"""
## 0. 준비

수업 자료(예시 음원, 도우미 코드)를 받고, 모델 가중치를 받을 폴더를 정합니다. 가중치는 모두 `models` 폴더 하나에 들어갑니다.
""")
code(BOOTSTRAP)


# ----------------------------------------------------------------------------------------------
md(r"""
## 1. 설치

Colab 기본 환경(Python 3.12, 사전 설치된 torch·numpy)을 그대로 쓰고, YourMT3+가 요구하는 패키지만 추가합니다. `transformers`는 모델 코드가 내부 모듈을 직접 가져다 쓰므로 Space와 같은 버전으로 고정합니다.
마지막 줄에 `설치 OK`가 찍히면 됩니다.
""")
code(r"""
import os, sys, importlib

if os.environ.get("CTAI_SKIP_INSTALL") != "1":      # 로컬 테스트용 스위치 (Colab에서는 무시)
    %pip install -q -r {REQ}/02_yourmt3.txt          # 목록: colab/requirements/02_yourmt3.txt

import numpy, torch, torchaudio, transformers, lightning, librosa, mido, einops, huggingface_hub
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | numpy {numpy.__version__} | transformers {transformers.__version__} | device: {DEVICE}")
print("설치 OK")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 2. 음악 고르기

세 가지 중 하나를 고릅니다. 모델은 앞 **60 초**만 사용합니다 (수업 시간 절약).

- **예제** — 수업 저장소 `colab/audio/`의 예시 음원. `band_lets_go_fishin_60s.wav` (Karissa Hobbs, *Let's Go Fishin'*, CC BY-NC 3.0, 보컬+밴드) / `jazz_vibe_ace_60s.wav` (Kevin MacLeod, *Vibe Ace*, CC BY 3.0)
- **업로드** — 내 wav/mp3 파일
- **YouTube URL** — `yt-dlp`로 음원만 내려받음. ※ 수업 내 학습 목적으로만 사용합니다.

라디오 버튼을 바꾼 뒤에는 **다음 셀(2-2)을 다시 실행**하세요.
""")
code(r"""
# 2-1. 소스 선택 위젯
import os, sys
from pathlib import Path
import ipywidgets as W
from IPython.display import display

WORK = os.environ.get("CTAI_WORK", "/content/work" if IN_COLAB else os.path.expanduser("~/ctai_work"))   # 임시 작업 폴더
Path(WORK).mkdir(parents=True, exist_ok=True)

EXAMPLES = {   # 수업 저장소 colab/audio/의 예시 음원. 파일명: (설명, 미사용)
    "band_lets_go_fishin_60s.wav": ("Karissa Hobbs – Let's Go Fishin' (CC BY-NC 3.0, 보컬+밴드)", None),
    "jazz_vibe_ace_60s.wav":       ("Kevin MacLeod – Vibe Ace (CC BY 3.0, 재즈 콤보)",           None),
}
src_w  = W.RadioButtons(options=[("예제", "example"), ("내 파일 업로드", "upload"), ("YouTube URL", "youtube")],
                        value="example", description="소스")
ex_w   = W.Dropdown(options=[(f"{k} — {v[0]}", k) for k, v in EXAMPLES.items()], description="예제")
url_w  = W.Text(value="", placeholder="https://www.youtube.com/watch?v=...", description="YouTube", layout=W.Layout(width="60%"))
display(W.VBox([src_w, ex_w, url_w]))
print("※ YouTube 음원은 수업 내 학습 목적으로만 사용합니다.")
print(f"audio dir: {AUDIO_DIR}\nwork dir : {WORK}\nout dir  : {OUT_DIR}")
""")
code(r"""
# 2-2. 선택한 소스를 읽어 60 초로 자르고 파형 + 재생
import subprocess, urllib.request
import numpy as np, librosa, soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio

MAX_SEC = 60
SR_LISTEN = 44100

def fetch_example(name):
    p = Path(AUDIO_DIR) / name
    assert p.exists(), f"예시 음원을 찾지 못했습니다: {p}. 맨 위 0. 준비 셀을 다시 실행하세요."
    return p

if src_w.value == "example":
    src_path = fetch_example(ex_w.value)
elif src_w.value == "upload":
    from google.colab import files
    up = files.upload()                          # 파일 선택 창
    name = next(iter(up)); src_path = Path(WORK) / name
    src_path.write_bytes(up[name])
else:
    assert url_w.value.strip(), "YouTube URL을 입력하세요"
    src_path = Path(WORK) / "youtube_audio.wav"
    subprocess.run(["yt-dlp", "-q", "-x", "--audio-format", "wav", "-o", str(src_path.with_suffix(".%(ext)s")),
                    url_w.value.strip()], check=True)

y_listen, _ = librosa.load(str(src_path), sr=SR_LISTEN, mono=True, duration=MAX_SEC)
DUR = len(y_listen) / SR_LISTEN
INPUT_WAV = Path(WORK) / "input_60s.wav"
sf.write(INPUT_WAV, y_listen, SR_LISTEN)
print(f"{src_path.name}: {DUR:.1f} s (앞 {MAX_SEC} s), mono {SR_LISTEN} Hz → {INPUT_WAV}")

# 과정 데모 페이지와 같은 어두운 스타일
PAL = dict(bg="#161616", fg="#e6e6e6", grid="#2b2b2b", blue="#58c4dd", yellow="#f5d547", red="#fc6255",
           green="#83c167", teal="#5cd0b3", purple="#b19cd9", brown="#c48a5a")
def dark_fig(*a, **k):
    fig, axes = plt.subplots(*a, **k)
    fig.patch.set_facecolor(PAL["bg"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#111111"); ax.tick_params(colors=PAL["fg"], labelsize=8)
        for s in ax.spines.values(): s.set_color(PAL["grid"])
        ax.xaxis.label.set_color(PAL["fg"]); ax.yaxis.label.set_color(PAL["fg"]); ax.title.set_color(PAL["fg"])
    return fig, axes

fig, ax = dark_fig(figsize=(12, 2.2))
t = np.arange(len(y_listen)) / SR_LISTEN
ax.plot(t, y_listen, color=PAL["blue"], lw=0.4); ax.set_xlim(0, DUR); ax.set_xlabel("time (s)"); ax.set_title("input waveform")
plt.tight_layout(); plt.show()
display(Audio(y_listen, rate=SR_LISTEN))
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 3. 모델 불러오기

YourMT3+의 코드와 체크포인트는 Hugging Face Space(`mimbres/YourMT3`)에 들어 있습니다. 코드(수 MB)와 체크포인트 `YPTF.MoE+Multi (noPS)`(561 MB)를 내려받고 파라미터 수를 세어 봅니다.
""")
code(r"""
import io, time, contextlib
from huggingface_hub import snapshot_download, hf_hub_download

SPACE_REV = "5e66c1ea173a8186e0d20432b841d3180cc015b5"        # 2025-01-31 Space 버전 고정
CKPT = "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"
YMT3_DIR = os.environ.get("CTAI_YMT3_DIR", f"{MODELS_DIR}/YourMT3")      # 코드와 체크포인트를 모델 폴더에

t0 = time.time()
if not os.path.exists(f"{YMT3_DIR}/{CKPT}"):
    snapshot_download("mimbres/YourMT3", repo_type="space", revision=SPACE_REV, local_dir=YMT3_DIR,
                      ignore_patterns=["*.ckpt", "examples/*", "mid/*"])
    hf_hub_download("mimbres/YourMT3", repo_type="space", revision=SPACE_REV, filename=CKPT, local_dir=YMT3_DIR)
print(f"코드+체크포인트 준비: {time.time()-t0:.0f} s → {YMT3_DIR}")

# Space 코드를 import 경로에 추가 (학습용 Lightning/wandb 코드가 같이 들어 있어 경고가 많이 뜨므로 숨김)
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)                     # torch.cuda.amp.autocast 경고 (무해)
warnings.filterwarnings("ignore", module="pytorch_lightning|lightning")
for p in (YMT3_DIR, f"{YMT3_DIR}/amt/src"):
    if p not in sys.path: sys.path.insert(0, p)
import config.config as ymt3_config
ymt3_config.shared_cfg["WANDB"]["save_dir"] = f"{YMT3_DIR}/amt/logs"    # 상대경로 → 절대경로
from model_helper import load_model_checkpoint
from utils.audio import slice_padded_array
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.note2event import mix_notes
from utils.utils import write_model_output_as_midi
for name in ("pytorch_lightning", "lightning.pytorch", "lightning", "lightning.fabric"):   # import 뒤에 설정해야 효과 있음
    logging.getLogger(name).setLevel(logging.ERROR)                           # 추론에는 쓰지 않는 Trainer 안내문 숨김

# Space app.py 의 "YPTF.MoE+Multi (noPS)" 설정 그대로
MODEL_ARGS = ["mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt", "-p", "2024",
              "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
              "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
              "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
              "-pr", "16" if DEVICE == "cuda" else "32"]
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():    # 긴 설정 출력 숨김
    warnings.simplefilter("ignore")
    model = load_model_checkpoint(args=MODEL_ARGS, device="cpu")
model = model.to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
SR_MODEL = model.audio_cfg["sample_rate"]; SEG = model.audio_cfg["input_frames"]
print(f"모델 로드 {time.time()-t0:.1f} s | 파라미터 {n_params/1e6:.1f} M | 입력 {SR_MODEL} Hz, {SEG/SR_MODEL:.3f} s 조각 단위 | device {DEVICE}")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 4. 추론

오디오를 16 kHz 모노로 바꾸고 2.048 초 조각으로 잘라 모델에 넣으면, 조각마다 토큰(이벤트) 열이 나옵니다. 토큰 → 음표 이벤트 → 음표로 되돌린 뒤 하나의 음표 목록으로 합칩니다.
**실시간 계수(RTF)** = 처리 시간 ÷ 음악 길이. 1보다 작으면 실시간보다 빠른 것.
""")
code(r"""
from collections import Counter

y16, _ = librosa.load(str(INPUT_WAV), sr=SR_MODEL, mono=True)
segments = slice_padded_array(y16[None, :], SEG, SEG)                        # (n_seg, SEG), 끝은 0으로 채움
segments = torch.from_numpy(segments.astype("float32")).unsqueeze(1).to(DEVICE)   # (n_seg, 1, SEG)

t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
    pred_tokens, _ = model.inference_file(bsz=8, audio_segments=segments)
t_infer = time.time() - t0

# 토큰 → 음표 (디코더 채널 13개를 각각 풀고 합침)
n_ch = model.task_manager.num_decoding_channels
start_secs = [SEG * i / SR_MODEL for i in range(segments.shape[0])]
notes_per_ch, err = [], Counter()
for ch in range(n_ch):
    zipped, _, _ = model.task_manager.detokenize_list_batches([a[:, ch, :] for a in pred_tokens], start_secs, return_events=True)
    n_ch_notes, e = merge_zipped_note_events_and_ties_to_notes(zipped)
    notes_per_ch.append(n_ch_notes); err += e
raw_notes = mix_notes(notes_per_ch)
t_total = time.time() - t0

# dict 목록으로 정리 (program = GM 프로그램 번호, 드럼은 128)
inv = model.midi_output_inverse_vocab
notes = []
for n in raw_notes:
    if n.is_drum:
        prog, name = 128, "Drums"
    else:
        prog, name = inv.get(n.program, (n.program, f"Prg {n.program}"))
    notes.append(dict(onset=round(float(n.onset), 3), offset=round(float(n.offset), 3), pitch=int(n.pitch),
                      program=int(prog), instrument=name, is_drum=bool(n.is_drum)))
notes.sort(key=lambda d: (d["onset"], d["program"], d["pitch"]))

print(f"조각 {segments.shape[0]}개 | 모델 추론 {t_infer:.1f} s | 후처리 포함 {t_total:.1f} s | 음악 {DUR:.1f} s → RTF {t_total/DUR:.2f}")
print(f"음표 {len(notes)}개, 악기별:")
for name, c in Counter(d["instrument"] for d in notes).most_common():
    print(f"  {name:22s} {c:5d}")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 5. 보기 · 듣기

모델이 내놓는 악기 라벨 16종을 **5개 묶음**(drums / bass / guitar / vocals / keys_other)으로 색을 입혀 피아노롤로 그립니다.
그리고 음표를 **사인파로 다시 소리내어**(드럼은 노이즈 버스트) 원곡과 비교해 들어 봅니다 — 모델이 "들은" 것이 이것입니다.
""")
code(r"""
CLASS_ORDER = ["drums", "bass", "guitar", "vocals", "keys_other"]
CLASS_COLOR = dict(drums=PAL["red"], bass=PAL["purple"], guitar=PAL["yellow"], vocals=PAL["teal"], keys_other=PAL["blue"])
DRUM_NAMES = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 42: "hh closed", 44: "hh pedal", 46: "hh open",
              49: "crash", 51: "ride", 41: "tom lo", 45: "tom", 47: "tom", 48: "tom", 50: "tom hi"}

def coarse_class(d):
    if d["is_drum"]: return "drums"
    p = d["program"]
    if 32 <= p <= 39: return "bass"
    if 24 <= p <= 31: return "guitar"
    if p in (100, 101): return "vocals"
    return "keys_other"          # piano, organ, strings, brass, reed, synth, ...

for d in notes: d["cls"] = coarse_class(d)

def plot_piano_roll(notes, title="YourMT3+ piano roll", t_max=None):
    t_max = t_max or DUR
    pitched = [d for d in notes if not d["is_drum"]]; drums = [d for d in notes if d["is_drum"]]
    fig, (ax, axd) = dark_fig(2, 1, figsize=(13, 6), sharex=True, gridspec_kw=dict(height_ratios=[4, 1.2]))
    for cls in CLASS_ORDER:
        sel = [d for d in pitched if d["cls"] == cls]
        for d in sel:
            ax.add_patch(plt.Rectangle((d["onset"], d["pitch"] - 0.4), max(d["offset"] - d["onset"], 0.03), 0.8,
                                           color=CLASS_COLOR[cls], lw=0))
    if pitched:
        lo = min(d["pitch"] for d in pitched) - 2; hi = max(d["pitch"] for d in pitched) + 2
        ax.set_ylim(lo, hi)
    ax.set_xlim(0, t_max); ax.set_ylabel("MIDI pitch"); ax.set_title(title)
    ax.grid(color=PAL["grid"], lw=0.5)
    ax.legend(handles=[plt.Line2D([], [], color=CLASS_COLOR[c], lw=6, label=c) for c in CLASS_ORDER],
              loc="upper right", fontsize=8, facecolor="#1f1f1f", labelcolor=PAL["fg"], edgecolor=PAL["grid"])
    if drums:
        drum_pitches = sorted({d["pitch"] for d in drums})
        ypos = {p: i for i, p in enumerate(drum_pitches)}
        axd.scatter([d["onset"] for d in drums], [ypos[d["pitch"]] for d in drums], s=10, color=CLASS_COLOR["drums"], marker="|", lw=1.5)
        axd.set_yticks(range(len(drum_pitches))); axd.set_yticklabels([f"{p} {DRUM_NAMES.get(p, '')}" for p in drum_pitches], fontsize=7)
    axd.set_ylabel("drums"); axd.set_xlabel("time (s)"); axd.grid(color=PAL["grid"], lw=0.5)
    plt.tight_layout(); plt.show()

def midi_hz(p): return 440.0 * 2 ** ((p - 69) / 12)

def synth(notes, sr=22050, dur=None, drum_gain=0.5):
    # 음표 → 사인파(짧은 어택, 지수 감쇠), 드럼 → 노이즈 버스트. 모델 출력만으로 들을 수 있는 가장 단순한 합성
    dur = dur or DUR
    out = np.zeros(int(sr * (dur + 1.0)), dtype=np.float32)
    rng = np.random.default_rng(0)
    for d in notes:
        s = int(d["onset"] * sr)
        if d["is_drum"]:
            p = d["pitch"]
            L = int(sr * (0.25 if p in (35, 36) else 0.12 if p in (38, 40) else 0.05))
            t = np.arange(L) / sr
            burst = rng.standard_normal(L) * np.exp(-t * (12 if p in (38, 40) else 40))
            if p in (35, 36):                                       # kick: 저음 사인 '쿵' 추가
                burst = 0.3 * burst + np.sin(2 * np.pi * 70 * t) * np.exp(-t * 18)
            sig = drum_gain * burst
        else:
            L = max(int((d["offset"] - d["onset"]) * sr), int(0.05 * sr))
            t = np.arange(L) / sr
            env = np.minimum(1, t / 0.005) * np.exp(-t * 2.0)
            env[-int(0.02 * sr):] *= np.linspace(1, 0, int(0.02 * sr))       # 짧은 릴리즈
            f = midi_hz(d["pitch"])
            sig = 0.25 * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t))
        e = min(s + len(sig), len(out)); out[s:e] += sig[: e - s]
    peak = np.abs(out).max()
    return (out / peak * 0.9) if peak > 0 else out

plot_piano_roll(notes)
SR_SYN = 22050
resynth = synth(notes, sr=SR_SYN)
print("모델 출력 (사인파 재합성)"); display(Audio(resynth, rate=SR_SYN))
print("원곡");                     display(Audio(y_listen, rate=SR_LISTEN))
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 6. 손잡이 돌려보기

- **class** — 한 묶음만 골라 솔로로 듣기 (드럼만, 보컬만, …)
- **min_dur** — 이 길이(초)보다 짧은 음표는 버림. 채보에서 흔한 "잡음표"를 걸러내는 가장 단순한 후처리
- **mix_original** — 원곡을 작게 섞어 비교

> 생각해 볼 질문: 드럼만 솔로로 들었을 때 드럼이 아닌 소리가 드럼으로 잡힌 곳이 있나요? 보컬 선율은 어느 묶음으로 새어 나가나요? `min_dur`를 올리면 어떤 음표가 먼저 사라지는지, 그 음표들이 실제 연주에 있던 음인지 원곡과 비교해 보세요.
""")
code(r"""
def render(cls="all", min_dur=0.0, mix_original=0.0):
    sel = [d for d in notes if (cls == "all" or d["cls"] == cls) and (d["offset"] - d["onset"] >= min_dur or d["is_drum"])]
    print(f"{len(sel)} / {len(notes)} 음표")
    if not sel:
        return
    plot_piano_roll(sel, title=f"class={cls}, min_dur={min_dur:.2f}s")
    mix = synth(sel, sr=SR_SYN)
    if mix_original > 0:
        o = librosa.resample(y_listen, orig_sr=SR_LISTEN, target_sr=SR_SYN)
        L = min(len(o), len(mix)); mix = mix[:L] + mix_original * o[:L]
    display(Audio(mix, rate=SR_SYN))

cls_w = W.Dropdown(options=["all"] + CLASS_ORDER, value="all", description="class")
dur_w = W.FloatSlider(min=0.0, max=0.5, step=0.02, value=0.0, description="min_dur (s)", continuous_update=False)
mix_w = W.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.0, description="mix_original", continuous_update=False)
out_w = W.Output()
def _on_change(_):
    with out_w:
        out_w.clear_output(wait=True)
        render(cls_w.value, dur_w.value, mix_w.value)
for w in (cls_w, dur_w, mix_w):
    w.observe(_on_change, names="value")
display(W.VBox([cls_w, dur_w, mix_w]), out_w)
render(cls_w.value, dur_w.value, mix_w.value)        # 처음 한 번은 기본값으로 바로 실행 (위젯을 바꾸면 위 out_w 에 다시 그려짐)
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 7. 저장

멀티트랙 MIDI(악기별 트랙, 드럼은 채널 10)와 음표 JSON을 결과 폴더(`outputs`)에 씁니다. Colab에서는 바로 내려받기도 합니다.
MIDI는 GarageBand / Logic / Ableton / MuseScore 등에서 열립니다.
""")
code(r"""
import json, shutil

stem = f"{src_path.stem}_yourmt3"
write_model_output_as_midi(raw_notes, WORK, stem, model.midi_output_inverse_vocab)   # WORK/model_output/<stem>.mid
midi_path = Path(OUT_DIR) / f"{stem}.mid"
shutil.move(Path(WORK) / "model_output" / f"{stem}.mid", midi_path)
json_path = Path(OUT_DIR) / f"{stem}.notes.json"
json_path.write_text(json.dumps({"meta": {"source": str(src_path), "duration_sec": round(DUR, 2), "model": "YourMT3+ YPTF.MoE+Multi (noPS)",
                                          "device": DEVICE, "runtime_sec": round(t_total, 1), "num_notes": len(notes)},
                                 "notes": notes}, indent=1, ensure_ascii=False))
print("saved:", midi_path, "\n       ", json_path)
if IN_COLAB and not str(OUT_DIR).startswith("/content/drive"):
    from google.colab import files
    files.download(str(midi_path)); files.download(str(json_path))
""")

nb.cells = cells
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
