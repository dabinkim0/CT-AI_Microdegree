"""Builds colab/notebooks/03_allin1_beat_structure.ipynb (outputs stripped)."""
import nbformat as nbf
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import BOOTSTRAP
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "03_allin1_beat_structure.ipynb"
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
# 03 · All-In-One (allin1) — 비트 · 다운비트 · 곡 구조 읽어내기

**All-In-One Music Structure Analyzer** (Kim & Nam, WASPAA 2023): 음악을 Demucs로 4개 스템(bass/drums/other/vocals)으로 나눈 뒤, 스템별 스펙트로그램에 **이웃 어텐션(NATTEN)** 을 걸어 BPM · 비트 · 다운비트 · 구간 경계 · 구간 라벨(verse/chorus/…)을 한 번에 내놓는 모델.

| | |
|---|---|
| 입력 → 출력 | 오디오(wav) → `bpm`, `beats`(초), `downbeats`(초), `beat_positions`(마디 안 위치 1~4), `segments`(start, end, label) + 프레임별 확률 곡선(100 fps) |
| 모델 | `harmonix-all` — 8-fold 앙상블(각 약 2.6 M 파라미터) + Demucs `htdemucs`(분리용) |
| 논문 / 코드 | [arXiv:2307.16425](https://arxiv.org/abs/2307.16425) · [github.com/mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) |
| 예상 시간 (Colab T4) | 설치 ≈ 1~2분 (madmom 빌드) · 60 초 음악 분석 ≈ 30 초 |
| GPU | **T4 권장**, 없어도 됨. 이웃 어텐션은 NATTEN 대신 저장소의 `natten_torch.py`(순수 PyTorch)로 돌리므로 CPU·맥(MPS)에서도 allin1이 돕니다. 설치가 실패하면 **madmom 폴백**(비트·다운비트만, 구간 없음)으로 넘어감 |

진행 순서: 준비 → 설치 → 음악 고르기 → 모델 불러오기 → 분석 → 구조 그림 보기·클릭 듣기 → 손잡이 돌려보기 → 저장
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

이 모델은 설치가 까다로운 편이라 순서가 중요합니다.

1. **madmom** — 비트 트래킹 후처리. PyPI 판이 깨져 있어 GitHub에서 직접 빌드합니다. torch와 numpy는 Colab 기본값을 그대로 씁니다.
2. **allin1 1.1.0**과 Demucs 등 의존 패키지. allin1은 의존성 없이 깔아서, 맥에서 NATTEN 소스 빌드를 끌어오지 않게 합니다.
3. **이웃 어텐션** — allin1은 NATTEN 0.17을 쓰는데, 이 버전은 Python 3.12·CUDA용 휠만 있습니다. 대신 같은 계산을 순수 PyTorch로 옮긴 `colab/natten_torch.py`를 `natten`으로 등록합니다(NATTEN 0.17.5와 결과가 같음을 확인).

마지막 줄에 `설치 OK`가 찍히면 됩니다. `ALLIN1_OK = False`로 나오면 아래 셀들은 자동으로 madmom 폴백으로 동작합니다.
""")
code(r"""
import os, sys
from pathlib import Path

if os.environ.get("CTAI_SKIP_INSTALL") != "1":      # 로컬 테스트용 스위치 (Colab에서는 무시)
    %pip install -q -r {REQ}/03_allin1_base.txt                  # cython (madmom 빌드용)
    %pip install -q --no-build-isolation -r {REQ}/03_madmom.txt  # madmom을 GitHub에서 빌드
    %pip install -q -r {REQ}/03_allin1.txt                       # allin1이 쓰는 패키지 (Demucs 등)
    %pip install -q --no-deps "allin1==1.1.0"                    # 맥에서 NATTEN 소스 빌드를 끌어오지 않도록 의존성 없이

import numpy, torch, madmom, librosa
import natten_torch; natten_torch.use()             # NATTEN 0.17 대신 순수 PyTorch 이웃 어텐션 (colab/natten_torch.py)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"   # mps: Apple Silicon 맥의 GPU

ALLIN1_OK = False
try:
    if os.environ.get("CTAI_FORCE_MADMOM") == "1":        # 로컬 테스트용: 폴백 경로 강제
        raise ImportError("CTAI_FORCE_MADMOM=1")
    import allin1
    from allin1.models import load_pretrained_model
    ALLIN1_OK = True
    print(f"allin1 1.1.0 / 이웃 어텐션: natten_torch (순수 PyTorch)")
except Exception as e:
    print("allin1을 불러올 수 없습니다 →", repr(e)[:160])
    print("→ madmom 폴백(비트·다운비트·BPM만)으로 진행합니다.")

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | numpy {numpy.__version__} | madmom {madmom.__version__} | device {DEVICE} | ALLIN1_OK = {ALLIN1_OK}")
print("설치 OK")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 2. 음악 고르기

세 가지 중 하나를 고릅니다. 모델은 앞 **60 초**만 사용합니다. (구간 라벨은 곡 전체를 넣을수록 정확하지만 수업 시간을 아끼기 위해 자릅니다.)

- **예제** — 수업 저장소 `colab/audio/`의 예시 음원. `band_lets_go_fishin_60s.wav` (Karissa Hobbs, *Let's Go Fishin'*, CC BY-NC 3.0, 보컬+밴드) / `jazz_vibe_ace_60s.wav` (Kevin MacLeod, *Vibe Ace*, CC BY 3.0)
- **업로드** — 내 wav/mp3 파일
- **YouTube URL** — `yt-dlp`로 음원만 내려받음. ※ 수업 내 학습 목적으로만 사용합니다.

라디오 버튼을 바꾼 뒤에는 **다음 셀(2-2)을 다시 실행**하세요.
""")
code(r"""
# 2-1. 경로 + 소스 선택 위젯
import ipywidgets as W
from IPython.display import display

WORK = os.environ.get("CTAI_WORK", "/content/work" if IN_COLAB else os.path.expanduser("~/ctai_work"))   # 임시 작업 폴더 (Demucs 분리 결과 등)
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
import urllib.request, time
import numpy as np, librosa, soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio

MAX_SEC = 60
SR = 44100                                   # allin1(Demucs)과 madmom 모두 44.1 kHz 기준

def fetch_example(name):
    p = Path(AUDIO_DIR) / name
    assert p.exists(), f"예시 음원을 찾지 못했습니다: {p}. 맨 위 0. 준비 셀을 다시 실행하세요."
    return p

if src_w.value == "example":
    src_path = fetch_example(ex_w.value)
elif src_w.value == "upload":
    src_path = ctai.upload(WORK)                 # Colab: 파일 선택 창 / 로컬: 파일 경로 입력
else:
    assert url_w.value.strip(), "YouTube URL을 입력하세요"
    src_path = Path(WORK) / "youtube_audio.wav"
    subprocess.run(["yt-dlp", "-q", "-x", "--audio-format", "wav", "-o", str(src_path.with_suffix(".%(ext)s")),
                    url_w.value.strip()], check=True)

y, _ = librosa.load(str(src_path), sr=SR, mono=True, duration=MAX_SEC)
DUR = len(y) / SR
INPUT_WAV = Path(WORK) / "input_60s.wav"
sf.write(INPUT_WAV, y, SR)
print(f"{src_path.name}: {DUR:.1f} s (앞 {MAX_SEC} s), mono {SR} Hz → {INPUT_WAV}")

# 과정 데모 페이지와 같은 어두운 스타일
PAL = dict(bg="#161616", fg="#e6e6e6", grid="#2b2b2b", blue="#58c4dd", yellow="#f5d547", red="#fc6255",
           green="#83c167", teal="#5cd0b3", purple="#b19cd9", brown="#c48a5a", gray="#9aa5b1")
def dark_fig(*a, **k):
    fig, axes = plt.subplots(*a, **k)
    fig.patch.set_facecolor(PAL["bg"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("#111111"); ax.tick_params(colors=PAL["fg"], labelsize=8)
        for s in ax.spines.values(): s.set_color(PAL["grid"])
        ax.xaxis.label.set_color(PAL["fg"]); ax.yaxis.label.set_color(PAL["fg"]); ax.title.set_color(PAL["fg"])
    return fig, axes

fig, ax = dark_fig(figsize=(12, 2.2))
ax.plot(np.arange(len(y)) / SR, y, color=PAL["blue"], lw=0.4); ax.set_xlim(0, DUR); ax.set_xlabel("time (s)"); ax.set_title("input waveform")
plt.tight_layout(); plt.show()
display(Audio(y, rate=SR))
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 3. 모델 불러오기

- **allin1 경로**: HF Hub `taejunkim/allinone`에서 8개 fold 체크포인트(각 1.4 MB)를 받아 앙상블로 묶고, 스템 분리용 Demucs `htdemucs`(약 80 MB)도 미리 받습니다.
- **madmom 폴백**: `RNNDownBeatProcessor`(BLSTM 8개 앙상블, 2016) + `DBNDownBeatTrackingProcessor`. 파라미터는 numpy 배열로 들어 있어 같은 식으로 셉니다.
""")
code(r"""
t0 = time.time()
if ALLIN1_OK:
    model = load_pretrained_model("harmonix-all", device=DEVICE)
    n_params = sum(p.numel() for m in model.models for p in m.parameters())
    print(f"allin1 harmonix-all: {len(model.models)}-fold 앙상블, 총 {n_params/1e6:.1f} M 파라미터 (fold당 {n_params/len(model.models)/1e6:.2f} M)")
    from demucs.pretrained import get_model as _get_demucs
    demucs_model = _get_demucs("htdemucs").to(DEVICE).eval()
    print(f"Demucs htdemucs: {sum(p.numel() for p in demucs_model.parameters())/1e6:.1f} M 파라미터 (분리용)")
    METHOD = "allin1 harmonix-all"
else:
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    rnn = RNNDownBeatProcessor()
    def _count_np(obj, seen=None):                      # madmom 모델은 numpy 배열 묶음 → 재귀적으로 원소 수 합산
        seen = seen if seen is not None else set()
        if id(obj) in seen: return 0
        seen.add(id(obj))
        if isinstance(obj, np.ndarray): return obj.size
        if isinstance(obj, (list, tuple)): return sum(_count_np(o, seen) for o in obj)
        if hasattr(obj, "__dict__"): return sum(_count_np(v, seen) for v in vars(obj).values())
        return 0
    ens = next(p for p in rnn.processors if type(p).__name__ == "NeuralNetworkEnsemble")
    n_params = _count_np(ens)
    print(f"madmom RNNDownBeatProcessor: BLSTM {len(ens.processors[0].processors)}개 앙상블, 총 {n_params/1e6:.2f} M 파라미터")
    METHOD = "madmom RNNDownBeat + DBN"
print(f"모델 준비 {time.time()-t0:.1f} s | device {DEVICE} | method: {METHOD}")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 4. 분석

allin1은 ① Demucs 분리 → ② 스템별 로그-멜 스펙트로그램(100 fps) → ③ 모델 → ④ madmom DBN으로 비트/다운비트 확정, 피크 피킹으로 구간 경계 확정 순으로 돕니다.
출력은 모두 **10 ms 격자**(100 fps) 위에 놓입니다. **실시간 계수(RTF)** = 처리 시간 ÷ 음악 길이.
""")
code(r"""
FPS = 100
t0 = time.time()
if ALLIN1_OK:
    # ① Demucs 분리를 여기서 직접 한다. allin1은 demucs.separate를 따로 실행하는데, 최신 torchaudio는 wav 저장에 torchcodec이 필요하다.
    from demucs.apply import apply_model
    x = torch.tensor(np.stack([y, y]), dtype=torch.float32)          # (2, T) 44.1 kHz
    ref = x.mean(0); xn = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        stems = apply_model(demucs_model, xn[None].to(DEVICE), device=DEVICE, shifts=1, split=True, overlap=0.25, progress=False)[0].cpu()
    stems = stems * (ref.std() + 1e-8) + ref.mean()
    stem_dir = Path(WORK) / "demix" / "htdemucs" / INPUT_WAV.stem     # allin1이 찾는 위치: 있으면 분리를 건너뜀
    stem_dir.mkdir(parents=True, exist_ok=True)
    for name, s in zip(demucs_model.sources, stems):
        sf.write(stem_dir / f"{name}.wav", s.numpy().T, SR, subtype="PCM_16")
    r = allin1.analyze(str(INPUT_WAV), device=DEVICE, include_activations=True, overwrite=True,
                       demix_dir=f"{WORK}/demix", spec_dir=f"{WORK}/spec", multiprocess=False)
    result = dict(method=METHOD, bpm=int(r.bpm), beats=[float(b) for b in r.beats], downbeats=[float(b) for b in r.downbeats],
                  beat_positions=[int(p) for p in r.beat_positions],
                  segments=[dict(start=float(s.start), end=float(s.end), label=s.label) for s in r.segments])
    activ = {k: np.asarray(v) for k, v in r.activations.items()}        # beat, downbeat, segment: (T,), label: (10, T)
else:
    act = rnn(y)                                                        # (T, 2): P(beat), P(downbeat) @100 fps
    bd = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS)(act)   # (N, 2): time, position
    beats = bd[:, 0]; pos = bd[:, 1].astype(int)
    result = dict(method=METHOD, bpm=int(round(60 / np.median(np.diff(beats)))) if len(beats) > 2 else 0,
                  beats=[float(b) for b in beats], downbeats=[float(b) for b in beats[pos == 1]],
                  beat_positions=[int(p) for p in pos], segments=[])
    activ = dict(beat=act[:, 0], downbeat=act[:, 1])
t_total = time.time() - t0

print(f"처리 {t_total:.1f} s | 음악 {DUR:.1f} s → RTF {t_total/DUR:.2f} | method {METHOD}")
print(f"BPM {result['bpm']} | 비트 {len(result['beats'])}개 | 다운비트 {len(result['downbeats'])}개 | 박자: {max(result['beat_positions'], default=0)}/4")
print("beats[:8]     :", result["beats"][:8])
print("downbeats[:4] :", result["downbeats"][:4])
if result["segments"]:
    for s in result["segments"]:
        print(f"  {s['start']:6.2f} – {s['end']:6.2f} s  {s['label']}")
else:
    print("segments: (madmom 폴백에서는 구간 라벨 없음)")
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 5. 보기 · 듣기

위: 파형 위에 **구간 라벨 색 띠**, 비트(짧은 눈금)와 다운비트(긴 노란 눈금).
아래: 모델이 프레임마다 내놓은 **비트/다운비트 확률 곡선**(100 fps) — DBN 후처리가 이 곡선에서 규칙적인 격자를 골라낸 것이 위의 눈금입니다.
그리고 비트마다 **클릭 소리**를 합성해 원곡 위에 얹어 들어 봅니다 (다운비트는 낮고 긴 클릭).
""")
code(r"""
LABEL_COLOR = dict(start=PAL["gray"], end=PAL["gray"], intro=PAL["teal"], outro=PAL["brown"], **{"break": PAL["purple"]},
                   bridge=PAL["green"], inst=PAL["blue"], solo=PAL["red"], verse=PAL["blue"], chorus=PAL["yellow"])
HARMONIX_LABELS = ["start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"]

def plot_structure(result, activ, title=None, show_labels=False):
    rows = 3 if (show_labels and "label" in activ) else 2
    hr = [3, 1.4, 1.6][:rows]
    fig, axes = dark_fig(rows, 1, figsize=(13, 2.2 * rows + 1), sharex=True, gridspec_kw=dict(height_ratios=hr))
    ax, ax2 = axes[0], axes[1]
    t = np.arange(len(y)) / SR
    ax.plot(t, y, color="#777777", lw=0.4, zorder=1)
    for s in result["segments"]:
        ax.axvspan(s["start"], s["end"], ymin=0.86, ymax=1.0, color=LABEL_COLOR.get(s["label"], PAL["gray"]), alpha=0.9, lw=0)
        if s["end"] - s["start"] > 1.5:
            ax.text((s["start"] + s["end"]) / 2, 0.93, s["label"], transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=8, color="#111111", fontweight="bold")
    ax.vlines(result["beats"], 0, 0.12, transform=ax.get_xaxis_transform(), color=PAL["blue"], lw=1)
    ax.vlines(result["downbeats"], 0, 0.86, transform=ax.get_xaxis_transform(), color=PAL["yellow"], lw=1.2, alpha=0.9)
    ax.set_xlim(0, DUR); ax.set_yticks([]); ax.set_ylabel("waveform")
    ax.set_title(title or f"{result['method']} — BPM {result['bpm']}, {len(result['beats'])} beats, {len(result['downbeats'])} downbeats")
    ax.legend(handles=[plt.Line2D([], [], color=PAL["blue"], lw=2, label="beat"), plt.Line2D([], [], color=PAL["yellow"], lw=2, label="downbeat")],
              loc="upper right", fontsize=8, facecolor="#1f1f1f", labelcolor=PAL["fg"], edgecolor=PAL["grid"])
    ta = np.arange(len(activ["beat"])) / FPS
    ax2.plot(ta, activ["beat"], color=PAL["blue"], lw=0.8, label="P(beat)")
    ax2.plot(ta, activ["downbeat"], color=PAL["yellow"], lw=0.8, label="P(downbeat)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("activation"); ax2.grid(color=PAL["grid"], lw=0.5)
    ax2.legend(loc="upper right", fontsize=8, facecolor="#1f1f1f", labelcolor=PAL["fg"], edgecolor=PAL["grid"])
    if rows == 3:
        ax3 = axes[2]; L = activ["label"]
        ax3.stackplot(np.arange(L.shape[1]) / FPS, L, labels=HARMONIX_LABELS, colors=[LABEL_COLOR[l] for l in HARMONIX_LABELS], lw=0)
        ax3.set_ylim(0, 1); ax3.set_ylabel("P(label)")
        ax3.legend(loc="upper right", fontsize=7, ncol=5, facecolor="#1f1f1f", labelcolor=PAL["fg"], edgecolor=PAL["grid"])
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout(); plt.show()

def click_track(result, sr=SR, mode="beats+downbeats", gain=0.8):
    # 비트 → 1500 Hz 25 ms 클릭, 다운비트 → 700 Hz 60 ms (더 낮고 길게)
    out = np.zeros(int(sr * (DUR + 0.5)), dtype=np.float32)
    def put(t_sec, f, L_sec, amp):
        s = int(t_sec * sr); L = int(L_sec * sr); tt = np.arange(L) / sr
        sig = amp * np.sin(2 * np.pi * f * tt) * np.exp(-tt * (8 / L_sec))
        e = min(s + L, len(out)); out[s:e] += sig[: e - s]
    db = set(np.round(result["downbeats"], 3))
    for b in result["beats"]:
        is_db = round(b, 3) in db
        if is_db and mode != "beats only":
            put(b, 700, 0.06, gain)
        elif not is_db and mode != "downbeats only":
            put(b, 1500, 0.025, gain * 0.7)
    return out

plot_structure(result, activ)
clicks = click_track(result)
mix = 0.5 * y + clicks[: len(y)]
print("원곡 + 클릭 (비트: 높은 소리, 다운비트: 낮고 긴 소리)"); display(Audio(mix / np.abs(mix).max() * 0.9, rate=SR))
print("원곡");                                                   display(Audio(y, rate=SR))
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 6. 손잡이 돌려보기

- **click_mode** — 비트+다운비트 / 다운비트만 / 비트만 들어보기
- **orig_mix** — 원곡 음량 (0이면 클릭만)
- **show_labels** — 구간 라벨 10종의 프레임별 확률을 한 장 더 그림 (allin1 경로에서만)

> 생각해 볼 질문: 다운비트만 듣고 손으로 "하나" 박자를 세어 보세요. 모델의 "하나"와 여러분의 "하나"가 어긋나는 구간이 있나요? 그 구간의 연주(드럼 필, 브레이크, 템포 흔들림)에 무슨 일이 있는지, 그리고 BPM이 두 배/절반으로 나올 수 있는 이유를 생각해 보세요. 구간 라벨이 바뀌는 지점은 여러분이 느끼는 곡의 전환점과 같나요?
""")
code(r"""
def render(click_mode="beats+downbeats", orig_mix=0.5, show_labels=False):
    plot_structure(result, activ, show_labels=show_labels)
    c = click_track(result, mode=click_mode)
    m = orig_mix * y + c[: len(y)]
    peak = np.abs(m).max()
    display(Audio(m / peak * 0.9 if peak > 0 else m, rate=SR))

mode_w = W.Dropdown(options=["beats+downbeats", "downbeats only", "beats only"], value="beats+downbeats", description="click_mode")
mix_w  = W.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.5, description="orig_mix", continuous_update=False)
lab_w  = W.Checkbox(value=False, description="show_labels (allin1 only)")
out_w  = W.Output()
def _on_change(_):
    with out_w:
        out_w.clear_output(wait=True)
        render(mode_w.value, mix_w.value, lab_w.value)
for w in (mode_w, mix_w, lab_w):
    w.observe(_on_change, names="value")
display(W.VBox([mode_w, mix_w, lab_w]), out_w)
render(mode_w.value, mix_w.value, lab_w.value)       # 처음 한 번은 기본값으로 바로 실행 (위젯을 바꾸면 위 out_w 에 다시 그려짐)
""")

# ----------------------------------------------------------------------------------------------
md(r"""
## 7. 저장

분석 결과 JSON(bpm, beats, downbeats, beat_positions, segments)과 클릭 트랙 wav를 결과 폴더(`outputs`)에 씁니다.
JSON은 영상 편집·비주얼 연동에서 "비트마다 컷" 같은 작업의 입력으로 바로 쓸 수 있는 형태입니다.
""")
code(r"""
import json

stem = f"{src_path.stem}_allin1" if ALLIN1_OK else f"{src_path.stem}_madmom"
json_path = Path(OUT_DIR) / f"{stem}.json"
json_path.write_text(json.dumps({"meta": {"source": str(src_path), "duration_sec": round(DUR, 2), "method": METHOD,
                                          "device": DEVICE, "runtime_sec": round(t_total, 1), "fps": FPS}, **result}, indent=1, ensure_ascii=False))
click_path = Path(OUT_DIR) / f"{stem}_clicks.wav"
sf.write(click_path, (0.5 * y + clicks[: len(y)]) / max(np.abs(0.5 * y + clicks[: len(y)]).max(), 1e-6) * 0.9, SR)
print("saved:", json_path, "\n       ", click_path)
if IN_COLAB and not str(OUT_DIR).startswith("/content/drive"):
    from google.colab import files
    files.download(str(json_path)); files.download(str(click_path))
""")

nb.cells = cells
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
