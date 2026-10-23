from _nb import *
cells = [
("md", """# 01 · Online AMT: 피아노 오디오 → MIDI (실시간)

**실시간 피아노 채보** (Kwon, Jeong & Nam. ISMIR 2020 LBD · PAR 모델의 전신). CNN이 mel-spectrogram에서 특징을 뽑고, 건반 88개마다 autoregressive LSTM이 5가지 상태(off · offset · sustain · onset · re-onset)를 매 32 ms 프레임마다 결정합니다. 앞 프레임의 예측을 다음 입력으로 쓰므로 **한 방향으로만** 처리할 수 있고, 그래서 실시간이 됩니다.

- 입력 → 출력: 피아노 오디오 → 프레임별 note 상태 → MIDI
- 소요 시간: 설치 1분 + 30초 곡 채보 CPU 약 10초
- GPU 불필요 (CPU 1스레드로도 실시간의 2배 속도)
- 논문: Kwon, Jeong & Nam, "Towards Efficient and Real-Time Piano Transcription Using Neural Autoregressive Models," IEEE/ACM TASLP 2024 · 코드: https://github.com/jdasam/online_amt"""),
("code", BOOTSTRAP),
("code", r'''#@title 설치
import subprocess, sys, os
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'librosa>=0.10', 'pretty_midi', 'mir_eval', 'soundfile', 'ipywidgets'], capture_output=True, text=True)
print(r.stderr[-600:] if r.returncode else '')
AMT_DIR = os.path.join(MODELS_DIR, 'online_amt')                   # 코드와 체크포인트를 모델 폴더 한 곳에
ctai.git_checkout('https://github.com/jdasam/online_amt', AMT_DIR)
# 최신 librosa에 맞춘 2줄 패치
p = os.path.join(AMT_DIR, 'autoregressive', 'mel.py'); s = open(p).read()
s = s.replace('pad_center(fft_window, filter_length)', 'pad_center(fft_window, size=filter_length)')
s = s.replace('mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)', 'mel(sr=sample_rate, n_fft=filter_length, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax, htk=True)')
open(p, 'w').write(s)
import torch, librosa; print('설치 OK · torch', torch.__version__, '· librosa', librosa.__version__)'''),
("code", AUDIO_PICK.format(EXAMPLES="{'피아노 (Scarlatti K.455, MAESTRO)': 'piano_scarlatti_k455_30s.wav'}",
    DEFAULT_EXAMPLE="피아노 (Scarlatti K.455, MAESTRO)", EXAMPLE_PARAM='["피아노 (Scarlatti K.455, MAESTRO)"]', MAX_SECONDS=30, TARGET_SR=16000, MONO=True)),
("code", r'''#@title 모델 로드 (체크포인트 179 MB, 처음 한 번만 내려받음)
import torch
ckpt = str(ctai.fetch(['https://media.githubusercontent.com/media/jdasam/online_amt/master/model-180000.pt',
                       'gdrive:12DnYJJ6YKpsoEkXI9fYUTTd4wOeY_rlB'],                  # 저자가 README에 올린 Google Drive 사본
                      os.path.join(AMT_DIR, 'model-180000.pt'), size=178804960))
sys.path.insert(0, AMT_DIR); sys.path.insert(0, ASSET_DIR)
from transcribe import load_model, OnlineTranscriber
from run_piano_amt import RawStateTranscriber, run_online, states_to_notes, velocity_proxy_from_audio, FPS
model = load_model(ckpt)
print('파라미터 수: %.1f M' % (sum(p.numel() for p in model.parameters()) / 1e6), '| 프레임률:', FPS, 'fps (hop 32 ms)')'''),
("code", r'''#@title 추론: 32 ms 청크로 실시간 루프 시뮬레이션
torch.set_num_threads(2)
t0 = time.time()
states, _total, step_s = run_online(model, mono)   # (frames, 88) 상태 id, 총 시간, 청크당 처리 시간(초)
dt = time.time() - t0; dur = len(mono) / sr
DELAY = 0.16   # 모델 구조상 160 ms (7프레임 버퍼 중앙) + 청크 32 ms + 연산 ≈ 200 ms
print('처리 %.1f s / 음원 %.1f s → 실시간의 %.1f배 | 청크당 %.1f ms (청크 길이 32 ms)' % (dt, dur, dur / dt, float(np.mean(step_s) * 1000)))
vel = velocity_proxy_from_audio(mono, len(states))
notes_raw = states_to_notes(states, FPS, velocity_proxy=vel, delay_sec=0.0)      # 보정 없음: 소리보다 0.2초 늦음
notes = states_to_notes(states, FPS, velocity_proxy=vel, delay_sec=DELAY)        # 지연 보정 (오프라인 분석용)
print('note 수:', len(notes))'''),
("code", r'''#@title 정답과 비교 (예시 음원은 연주자의 MIDI가 있음)
import pretty_midi, mir_eval
gt_path = os.path.join(AUDIO_DIR, 'piano_scarlatti_k455_30s.mid')
gt = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(gt_path).instruments for n in i.notes] if (SOURCE == '예시' and os.path.exists(gt_path)) else []
def f1(nts, ref):
    if not ref or not nts: return None
    a = np.array([[n['onset'], n['offset']] for n in nts]); p = np.array([librosa.midi_to_hz(n['pitch']) for n in nts])
    ra = np.array([[s, e] for s, e, _ in ref]); rp = np.array([librosa.midi_to_hz(m) for _, _, m in ref])
    return mir_eval.transcription.precision_recall_f1_overlap(ra, rp, a, p, offset_ratio=None)[2]
fa, fb = f1(notes_raw, gt), f1(notes, gt)
print('onset F1 (50 ms 허용) · 지연 보정 전: %s · 보정 후: %s' % ('%.2f' % fa if fa is not None else '정답 없음', '%.2f' % fb if fb is not None else '정답 없음'))'''),
("code", SINE_HELPERS),
("code", r'''#@title 듣기·보기: 피아노롤 + 사인파 합성
fig, ax = plt.subplots(figsize=(14, 4.5))
for s_, e_, p_ in gt: ax.plot([s_, e_], [p_, p_], color='#f5d547', lw=4, alpha=.5, solid_capstyle='butt')
for n in notes: ax.plot([n['onset'], n['offset']], [n['pitch'], n['pitch']], color='#58c4dd', lw=2, solid_capstyle='butt')
ax.set_xlabel('time (s)'); ax.set_ylabel('MIDI pitch'); ax.set_title('blue: model output (delay-compensated) · yellow: ground truth'); plt.tight_layout(); plt.show()
synth = synth_notes([(n['onset'], n['offset'], n['pitch']) for n in notes], sr=sr, duration=dur)
print('원음'); display(Audio(mono, rate=sr))
print('채보 결과 (사인파)'); display(Audio(synth, rate=sr))
print('원음(왼쪽) + 채보(오른쪽)'); display(Audio(np.stack([mono[:len(synth)], synth[:len(mono)]]), rate=sr))'''),
("md", """## 해 보기

1. **지연을 귀로 듣기**: 아래 셀은 지연 보정을 뺀 결과(`notes_raw`)를 원음과 겹쳐 재생합니다. 0.2초 늦게 따라오는 것이 들리나요? 실시간 비주얼에서는 이 지연을 어떻게 다루면 좋을까요?
2. **velocity와 offset**: 이 모델은 세기(velocity)를 출력하지 않아 음량으로 추정하고, 페달을 많이 쓰는 곡에서는 offset이 흔들립니다. 자기 피아노 음원을 올려 보고(`SOURCE='업로드'`), 어떤 음이 빠지거나 늘어지는지 찾아 보세요."""),
("code", r'''#@title 지연 보정 전/후 비교 재생
synth_raw = synth_notes([(n['onset'], n['offset'], n['pitch']) for n in notes_raw], sr=sr, duration=dur)
print('보정 전 (실시간 그대로, 0.2 s 늦음): 원음 L / 합성 R'); display(Audio(np.stack([mono[:len(synth_raw)], synth_raw[:len(mono)]]), rate=sr))
print('보정 후: 원음 L / 합성 R'); display(Audio(np.stack([mono[:len(synth)], synth[:len(mono)]]), rate=sr))'''),
("code", r'''#@title 저장: MIDI와 note JSON
import pretty_midi
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
for n in notes: inst.notes.append(pretty_midi.Note(int(n.get('velocity', 80)), int(n['pitch']), float(n['onset']), float(n['offset'])))
pm.instruments.append(inst); midi_out = os.path.join(OUT_DIR, 'online_amt.mid'); pm.write(midi_out)
json.dump(notes, open(os.path.join(OUT_DIR, 'online_amt_notes.json'), 'w')); print('저장:', midi_out)'''),
]
build("01_online_amt_piano_transcription.ipynb", cells, "01 Online AMT")
