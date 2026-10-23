from _nb import *
cells = [
("md", """# 04 · Demucs: 음악 오디오 → 6 스템

**Hybrid Transformer Demucs** (Rouard, Massa & Défossez, ICASSP 2023). 파형과 스펙트로그램 두 갈래 U-Net 사이에 Transformer를 둔 스템 분리 모델. `htdemucs_6s`는 vocals · drums · bass · guitar · piano · other 6개로 나눕니다.

- 입력 → 출력: 음악 오디오 → 6개의 오디오
- 소요 시간: 설치 2분 + 60초 곡 분리 T4 약 20초 (CPU 약 2분)
- 논문: https://arxiv.org/abs/2211.08553"""),
("code", BOOTSTRAP),
("code", r'''#@title 설치
%pip install -q -r {REQ}/04_demucs.txt
import demucs, torch; print('설치 OK · demucs', demucs.__version__, '· torch', torch.__version__, '· GPU:', torch.cuda.is_available())'''),
("code", AUDIO_PICK.format(EXAMPLES="{'밴드 (Let\\'s Go Fishin\\', 보컬+밴드)': 'band_lets_go_fishin_60s.wav', '재즈 (Vibe Ace)': 'jazz_vibe_ace_60s.wav'}",
    DEFAULT_EXAMPLE="밴드 (Let's Go Fishin', 보컬+밴드)", EXAMPLE_PARAM='["밴드 (Let\'s Go Fishin\', 보컬+밴드)", "재즈 (Vibe Ace)"]', MAX_SECONDS=60, TARGET_SR=44100, MONO=False)),
("code", r'''#@title 모델 로드
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model('htdemucs_6s').to(device).eval()
print('스템:', model.sources, '| 파라미터 수: %.1f M' % (sum(p.numel() for p in model.parameters()) / 1e6), '| device:', device)'''),
("code", r'''#@title 추론: 분리 실행
wav = torch.tensor(y if y.ndim == 2 else np.stack([y, y]), dtype=torch.float32)   # (2, T)
ref = wav.mean(0); wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
t0 = time.time()
with torch.no_grad():
    out = apply_model(model, wav_n[None].to(device), device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
out = out * (ref.std() + 1e-8) + ref.mean()
stems = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
dt = time.time() - t0; dur = wav.shape[1] / sr
print('처리 시간 %.1f s / 음원 %.1f s → 실시간의 %.1f배 속도' % (dt, dur, dur / dt))'''),
("code", r'''#@title 듣기·보기: 스템별 에너지와 재생
fps = 50; hop = sr // fps
fig, axes = plt.subplots(len(stems), 1, figsize=(14, 1.2 * len(stems) + 1), sharex=True)
colors = {'vocals': '#fc6255', 'drums': '#58c4dd', 'bass': '#c48a5a', 'guitar': '#83c167', 'piano': '#b19cd9', 'other': '#a9a9a9'}
for ax, (name, s) in zip(axes, stems.items()):
    m = s.mean(0); rms = librosa.feature.rms(y=m, frame_length=hop * 2, hop_length=hop)[0]; rms = rms / (rms.max() + 1e-9)
    ax.fill_between(np.arange(len(rms)) / fps, 0, rms, color=colors.get(name, '#888'), alpha=.8); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha='right')
axes[-1].set_xlabel('time (s)'); plt.tight_layout(); plt.show()
print('원음'); display(Audio(wav.numpy(), rate=sr))
for name, s in stems.items():
    print(name); display(Audio(s, rate=sr))'''),
("md", """## 해 보기

1. 아래 슬라이더로 스템을 섞어 **리믹스**를 만들어 보세요. 보컬만 빼면 MR(반주)이 됩니다.
2. 가장 분리가 **안 되는** 악기는 무엇인가요? (힌트: `other`와 `piano`를 혼자 들어 보기) 왜 그럴까요?"""),
("code", r'''#@title 리믹스: 스템 가중치 (0 = 끄기, 1 = 원래대로)
vocals = 1.0 #@param {type:"slider", min:0, max:1.5, step:0.1}
drums = 1.0  #@param {type:"slider", min:0, max:1.5, step:0.1}
bass = 1.0   #@param {type:"slider", min:0, max:1.5, step:0.1}
guitar = 1.0 #@param {type:"slider", min:0, max:1.5, step:0.1}
piano = 1.0  #@param {type:"slider", min:0, max:1.5, step:0.1}
other = 1.0  #@param {type:"slider", min:0, max:1.5, step:0.1}
w = dict(vocals=vocals, drums=drums, bass=bass, guitar=guitar, piano=piano, other=other)
remix = sum(w[k] * stems[k] for k in stems)
display(Audio(remix, rate=sr))'''),
("code", r'''#@title 저장: 스템 wav를 outputs/stems 폴더에
stem_dir = os.path.join(OUT_DIR, 'stems'); os.makedirs(stem_dir, exist_ok=True)
for name, s in stems.items(): sf.write(os.path.join(stem_dir, name + '.wav'), s.T, sr)
sf.write(os.path.join(stem_dir, 'vocals_20s_16k.wav'), librosa.resample(stems['vocals'].mean(0)[:20 * sr], orig_sr=sr, target_sr=16000), 16000)   # 05 Tone Transfer 입력용
print('저장:', sorted(os.listdir(stem_dir)))'''),
]
build("04_demucs_stem_separation.ipynb", cells, "04 Demucs")
