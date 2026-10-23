from _nb import *
cells = [
("md", """# 05 · DDSP Tone Transfer: 단선율 오디오 → 악기 소리

**DDSP (Differentiable Digital Signal Processing)** (Engel, Hantrakul, Gu & Roberts, ICLR 2020). 입력 소리에서 **음높이(f0)** 와 **음량(loudness)** 곡선 두 개만 뽑아, 그 곡선으로 미분 가능한 신디사이저(배음 합성기 + 필터 노이즈 + 리버브)를 제어합니다. 신디사이저의 "손가락"은 작은 GRU 디코더가 악기별 녹음 10여 분으로 학습한 것이라, 같은 곡선을 넣어도 Flute · Saxophone · Trumpet · Violin 각각의 음색이 나옵니다.

- 입력 → 출력: 단선율 오디오 (목소리, 트럼펫 등) → 같은 멜로디의 악기 오디오 4종
- 소요 시간: 설치 2분 + 10초 음원 기준 T4 약 1분 (CPU 약 3–4분: 피치 추출 CREPE가 느림)
- GPU: 선택 (T4 권장). 모델은 작아서(약 6 M 파라미터) CPU로도 됩니다.
- 논문: https://arxiv.org/abs/2001.04643 · 웹 데모: https://sites.research.google/tonetransfer"""),
("code", BOOTSTRAP),
("code", r'''#@title 설치 (약 2분)
import os, sys, subprocess, importlib.metadata as md
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # ddsp 3.7은 Keras 2 API를 씀 → TF 2.16+에서는 tf-keras로 대체
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # INFO 로그 숨김
def pip(*args):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], capture_output=True, text=True)
    if r.returncode: print('pip 경고:', r.stderr.strip().splitlines()[-1][:300])
    return r.returncode == 0
tfv = '.'.join(md.version('tensorflow').split('.')[:2])          # Colab에 깔린 TensorFlow와 같은 minor 버전의 tf-keras
pip('tf-keras==%s.*' % tfv, 'tensorflow-probability==0.25.0', 'gin-config==0.5.0', 'hmmlearn', 'mir_eval', 'note-seq', 'cloudml-hypertune', 'tensorflow-datasets', 'google-cloud-storage', 'pydub', 'resampy', 'ipywidgets')
if not pip('--no-build-isolation', 'crepe==0.0.16'):                 # setup.py가 pkg_resources를 써서 격리 빌드를 끔
    pip('setuptools<81'); pip('--no-build-isolation', 'crepe==0.0.16')   # 최신 setuptools에는 pkg_resources가 없음 → 한 번 내려서 재시도
pip('--no-deps', 'ddsp==3.7.0')                                     # 메타데이터가 TF<=2.11을 요구하지만 추론은 최신 TF에서도 됨
DDSP_OK = False
try:
    import tensorflow as tf, ddsp, ddsp.training, crepe, gin
    DDSP_OK = True
    print('설치 OK · ddsp', ddsp.__version__, '· tensorflow', tf.__version__, '· keras:', 'tf-keras (legacy)' if 'tf_keras' in tf.keras.__name__ else tf.keras.__name__, '· GPU:', bool(tf.config.list_physical_devices('GPU')))
except Exception as e:
    print('ddsp를 불러오지 못했습니다 → 아래 셀들은 대체 경로(f0·loudness 추출 + 사전 렌더 예시 + 웹 데모)로 진행합니다.\n   원인:', repr(e)[:300])
    print('설치 OK (대체 경로)')'''),
("code", r'''#@title 오디오 준비: 예시 트럼펫 / 04의 보컬 스템 / 마이크 녹음 / 업로드
import soundfile as sf, numpy as np, librosa
from IPython.display import Audio, display
import matplotlib.pyplot as plt

SOURCE = '예시 (트럼펫)'   #@param ["예시 (트럼펫)", "04 보컬 스템", "마이크 녹음", "업로드"]
RECORD_SECONDS = 8        #@param {type:"slider", min:3, max:15, step:1}
MAX_SECONDS = 10          #@param {type:"integer"}
TARGET_SR = 16000         # CREPE와 DDSP 체크포인트가 모두 16 kHz

def load_audio(path, sr=TARGET_SR, max_sec=MAX_SECONDS):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if max_sec: y = y[:int(max_sec * sr)]
    return y.astype(np.float32), sr

def record_mic(seconds, out='mic_input.wav'):
    """Colab 브라우저 마이크로 녹음 (JS MediaRecorder → webm → ffmpeg → 16 kHz wav)."""
    from google.colab import output
    from IPython.display import Javascript
    from base64 import b64decode
    display(Javascript("""
    const sleep = t => new Promise(r => setTimeout(r, t));
    const b2text = blob => new Promise(r => { const fr = new FileReader(); fr.onloadend = e => r(e.srcElement.result); fr.readAsDataURL(blob); });
    window.record = async (ms) => {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const rec = new MediaRecorder(stream); const chunks = [];
      rec.ondataavailable = e => chunks.push(e.data);
      rec.start(); await sleep(ms);
      return new Promise(r => { rec.onstop = async () => { stream.getTracks().forEach(t => t.stop()); r(await b2text(new Blob(chunks))); }; rec.stop(); });
    };"""))
    print('녹음 중... %d초 동안 한 음씩 끊지 말고 허밍/노래해 보세요 (단선율).' % seconds)
    data = output.eval_js('record(%d)' % (seconds * 1000))
    open('mic_input.webm', 'wb').write(b64decode(data.split(',')[1]))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'mic_input.webm', '-ac', '1', '-ar', str(TARGET_SR), out], check=True)
    return out

if SOURCE == '04 보컬 스템':
    audio_path = os.path.join(OUT_DIR, 'stems', 'vocals_20s_16k.wav')   # 04 노트북이 저장한 파일
    if not os.path.exists(audio_path): print('04의 보컬 스템이 없습니다. 04와 05를 모두 KEEP_IN_DRIVE를 켜고 실행하면 이어서 쓸 수 있습니다. 예시 트럼펫으로 대신합니다.'); audio_path = os.path.join(AUDIO_DIR, 'trumpet_solo_5s.wav')
elif SOURCE == '마이크 녹음':
    try: audio_path = record_mic(RECORD_SECONDS)
    except Exception as e: print('마이크 녹음은 Colab 브라우저에서만 됩니다 (%s). 예시 트럼펫으로 대신합니다.' % type(e).__name__); audio_path = os.path.join(AUDIO_DIR, 'trumpet_solo_5s.wav')
elif SOURCE == '업로드':
    from google.colab import files
    up = files.upload(); audio_path = list(up.keys())[0]
else:
    audio_path = os.path.join(AUDIO_DIR, 'trumpet_solo_5s.wav')

mono, sr = load_audio(audio_path)
print('파일:', audio_path, '| 길이: %.1f s | sr=%d' % (len(mono) / sr, sr))
plt.figure(figsize=(14, 2)); plt.plot(np.arange(len(mono)) / sr, mono, lw=.4, color='#58c4dd'); plt.xlabel('time (s)'); plt.title('waveform'); plt.tight_layout(); plt.show()
display(Audio(mono, rate=sr))'''),
("md", """## 1. 제어 신호 추출: 소리 → f0 곡선 + loudness 곡선

DDSP 모델이 입력에서 보는 것은 이 두 곡선뿐입니다. 음색(배음 구조, 숨소리)은 모두 버려지고, 디코더가 악기의 음색으로 다시 채웁니다.

- **f0 (음높이)**: CREPE (CNN 피치 추적기, 16 kHz) + Viterbi 스무딩. 함께 나오는 confidence가 낮은 프레임 = 무음·자음·숨.
- **loudness (음량)**: A-weighting(사람 귀의 감도 곡선)을 적용한 dB 단위 에너지.
- 프레임률 250 fps (hop 64 샘플 = 4 ms). 1초 음원 → 250개의 (f0, loudness) 쌍."""),
("code", r'''#@title 특징 추출: f0 (CREPE) + loudness @ 250 fps
FRAME_RATE = 250; HOP = TARGET_SR // FRAME_RATE
audio = mono[np.newaxis, :]                                   # (1, n_samples)

def compute_loudness_np(y, sr=TARGET_SR, frame_rate=FRAME_RATE, n_fft=512, range_db=120.0, ref_db=0.0):
    """ddsp 3.x의 spectral_ops.compute_loudness과 같은 식: A-weighted power를 주파수 평균 → dB."""
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=sr // frame_rate, center=True)) ** 2
    power = power * 10 ** (librosa.A_weighting(librosa.fft_frequencies(sr=sr, n_fft=n_fft))[:, None] / 10)
    loud = 10 * np.log10(np.maximum(power.mean(axis=0), 10 ** (-range_db / 10))) - ref_db
    return np.maximum(loud, -range_db).astype(np.float32)

t0 = time.time()
if DDSP_OK:
    feats = ddsp.training.metrics.compute_audio_features(audio, FRAME_RATE)
    feats = {k: (v.numpy() if hasattr(v, 'numpy') else np.asarray(v)) for k, v in feats.items()}
    feats['loudness_db'] = feats['loudness_db'].astype(np.float32)
    F0_METHOD = 'CREPE (full) + viterbi'
else:
    try:
        import crepe
        _, f0, conf, _ = crepe.predict(mono, sr, viterbi=True, step_size=1000 // FRAME_RATE, center=True, verbose=0)
        F0_METHOD = 'CREPE (full) + viterbi'
    except Exception:
        f0, vf, vp = librosa.pyin(mono, fmin=50, fmax=2000, sr=sr, frame_length=1024, hop_length=HOP)
        f0 = np.nan_to_num(f0); conf = vp; F0_METHOD = 'librosa.pyin'
    feats = {'audio': audio, 'f0_hz': f0.astype(np.float32), 'f0_confidence': conf.astype(np.float32), 'loudness_db': compute_loudness_np(mono)}
n = min(len(feats['f0_hz']), len(feats['loudness_db']), len(mono) // HOP)
for k in ('f0_hz', 'f0_confidence', 'loudness_db'): feats[k] = feats[k][:n]
feats['audio'] = audio[:, :n * HOP]
dt = time.time() - t0
print('f0: %s | 프레임 수 %d → %.0f fps | 추출 %.1f s (음원 %.1f s)' % (F0_METHOD, n, n / (len(mono) / sr), dt, len(mono) / sr))
voiced = feats['f0_confidence'] > 0.7
print('confidence>0.7 프레임 %.0f%% | f0 중앙값 %.0f Hz (MIDI %.1f) | loudness 평균 %.1f dB' % (100 * voiced.mean(), np.median(feats['f0_hz'][voiced]) if voiced.any() else 0, librosa.hz_to_midi(np.median(feats['f0_hz'][voiced])) if voiced.any() else 0, feats['loudness_db'].mean()))'''),
("code", r'''#@title 보기: f0 · loudness 곡선 (웹 데모 페이지의 두 그래프)
def plot_controls(f0_hz, loudness_db, conf=None, title='', ax=None, color='#fc6255'):
    t = np.arange(len(f0_hz)) / FRAME_RATE
    ax1, ax2 = ax
    f0 = f0_hz.copy(); f0[f0 <= 0] = np.nan
    if conf is not None: f0 = np.where(conf > 0.7, f0, np.nan); ax1.plot(t, np.where(conf > 0.7, np.nan, f0_hz), color=color, lw=.6, alpha=.25)
    ax1.plot(t, f0, color=color, lw=1.2); ax1.set_yscale('log'); ax1.set_ylabel('f0 (Hz)'); ax1.set_title(title or 'f0 (solid: confidence > 0.7, faint: unvoiced)')
    ax1.set_yticks([110, 220, 440, 880, 1760]); ax1.set_yticklabels(['A2 110', 'A3 220', 'A4 440', 'A5 880', 'A6 1760']); ax1.yaxis.set_minor_formatter(plt.NullFormatter()); ax1.grid(axis='y', alpha=.3)
    ax2.fill_between(t, -120, loudness_db, color='#58c4dd', alpha=.7); ax2.set_ylim(-100, 0); ax2.set_ylabel('loudness (dB)'); ax2.set_xlabel('time (s)')
fig, ax = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
plot_controls(feats['f0_hz'], feats['loudness_db'], feats['f0_confidence'], ax=ax); plt.tight_layout(); plt.show()'''),
("code", r'''#@title 듣기: 두 곡선만으로 사인파 재합성 (모델 없이, "제어 신호"가 무엇인지 확인)
def sine_from_controls(f0_hz, loudness_db, conf=None, sr=TARGET_SR, hop=HOP, gate_db=-70):
    f0 = np.repeat(f0_hz, hop); ld = np.repeat(loudness_db, hop)
    amp = 10 ** (np.clip(ld, gate_db, 0) / 20) * (ld > gate_db)
    if conf is not None: amp = amp * np.repeat(np.clip((conf - 0.5) / 0.3, 0, 1), hop)   # 무음·자음 구간은 끔
    amp = np.convolve(amp, np.ones(hop * 4) / (hop * 4), mode='same')
    y = np.sin(2 * np.pi * np.cumsum(f0) / sr) * amp
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)
sine = sine_from_controls(feats['f0_hz'], feats['loudness_db'], feats['f0_confidence'])
print('입력'); display(Audio(mono, rate=sr)); print('f0 + loudness → 사인파'); display(Audio(sine, rate=sr))'''),
("md", """## 2. 악기 4종으로 변환

체크포인트는 Google이 공개한 것(2021-07-08)으로, 악기당 약 58 MB입니다. 각 모델에는 학습 데이터의 **평균 음높이와 음량 분포**(`dataset_statistics.pkl`)가 함께 있어서, 입력 f0를 악기의 음역으로 옥타브 단위로 옮기고 음량을 악기의 분포에 맞춥니다 (공식 노트북의 "ADJUST" 블록과 같음)."""),
("code", r'''#@title 모델: 체크포인트 준비 + 추론 함수
INSTRUMENTS = {'violin': 'solo_violin_ckpt', 'flute': 'solo_flute_ckpt', 'trumpet': 'solo_trumpet_ckpt', 'tenor_saxophone': 'solo_tenor_saxophone_ckpt'}
CKPT_BASE = 'https://storage.googleapis.com/ddsp/models/timbre_transfer_colab/2021-07-08'
CKPT_ROOT = os.path.join(MODELS_DIR, 'ddsp')                        # 악기당 약 58 MB, 처음 한 번만 내려받음
EXAMPLE_DIR = os.path.join(ASSET_DIR, 'ddsp_examples')     # 미리 렌더한 예시 (DDSP가 안 깔릴 때 듣기용)

def fetch_ckpt(name):
    d = os.path.join(CKPT_ROOT, INSTRUMENTS[name]); os.makedirs(d, exist_ok=True)
    step = 40000 if name == 'violin' else 20000
    for f in ['ckpt-%d.data-00000-of-00001' % step, 'ckpt-%d.index' % step, 'dataset_statistics.pkl', 'operative_config-0.gin']:
        p = os.path.join(d, f)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            ctai.fetch('%s/%s/%s' % (CKPT_BASE, INSTRUMENTS[name], f), p)
    return d, os.path.join(d, 'ckpt-%d' % step)

if DDSP_OK:
    import pickle
    from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
    def shift_f0(f0_hz, octaves): return np.clip(f0_hz * 2.0 ** octaves, 0.0, librosa.midi_to_hz(110.0)).astype(np.float32)
    def adjust_features(feats, stats, note_threshold=1.0, conf_threshold=0.7, quiet=20.0, pitch_shift=0.0, loudness_shift=0.0):
        """공식 노트북 'Modify conditioning': 옥타브 맞춤 + loudness 분위수 매핑 + 노트 밖 구간 -20 dB."""
        mod = {k: v.copy() for k, v in feats.items()}; info = {}
        mask_on, note_on = detect_notes(feats['loudness_db'], feats['f0_confidence'], note_threshold, f0_confidence_threshold=conf_threshold)
        if np.any(mask_on):
            p_in = float(np.mean(ddsp.core.hz_to_midi(feats['f0_hz'])[mask_on])); d = (stats['mean_pitch'] - p_in) / 12.0
            octs = float(np.floor(d) if d > 1.5 else np.ceil(d))
            mod['f0_hz'] = shift_f0(mod['f0_hz'], octs)
            _, ld = fit_quantile_transform(feats['loudness_db'], mask_on, inv_quantile=stats['quantile_transform'])
            off = ~mask_on; ld[off] -= quiet * (1.0 - note_on[off][:, np.newaxis])
            ld = np.reshape(ld, feats['loudness_db'].shape)
            m_on = float(np.mean(ld[mask_on]))
            auto_gain = 0.0
            if m_on < -55:   # 짧거나 조용한 클립에서는 분위수 매핑이 과하게 낮춰 무음이 되기도 함 → 자동 보정
                auto_gain = -40.0 - m_on; ld = ld + auto_gain
            mod['loudness_db'] = ld.astype(np.float32)
            info = dict(input_mean_midi=p_in, target_mean_midi=float(stats['mean_pitch']), octave_shift=octs, note_on=float(mask_on.mean()), auto_loudness_gain_db=auto_gain)
        if pitch_shift: mod['f0_hz'] = shift_f0(mod['f0_hz'], pitch_shift)
        if loudness_shift: mod['loudness_db'] = mod['loudness_db'] + np.float32(loudness_shift)
        return mod, info
    def build_model(model_dir, ckpt, n_frames):
        gin.clear_config()
        with gin.unlock_config(): gin.parse_config_file(os.path.join(model_dir, 'operative_config-0.gin'), skip_unknown=True)
        hop = int(gin.query_parameter('Harmonic.n_samples') / gin.query_parameter('F0LoudnessPreprocessor.time_steps'))   # 64
        with gin.unlock_config():
            gin.parse_config(['Harmonic.n_samples = %d' % (n_frames * hop), 'FilteredNoise.n_samples = %d' % (n_frames * hop),
                              'F0LoudnessPreprocessor.time_steps = %d' % n_frames, 'oscillator_bank.use_angular_cumsum = True',
                              'F0LoudnessPreprocessor.compute_loudness = False'])   # 우리가 조정한 loudness를 그대로 쓰게 (기본값 True면 audio에서 다시 계산해 버림)
        model = ddsp.training.models.Autoencoder(); model.restore(ckpt, verbose=False)
        return model
    def tone_transfer(name, feats, pitch_shift=0.0, loudness_shift=0.0):
        model_dir, ckpt = fetch_ckpt(name)
        stats = pickle.load(open(os.path.join(model_dir, 'dataset_statistics.pkl'), 'rb'))
        mod, info = adjust_features(feats, stats, pitch_shift=pitch_shift, loudness_shift=loudness_shift)
        model = build_model(model_dir, ckpt, len(mod['f0_hz']))
        out = model.get_audio_from_outputs(model(mod, training=False)).numpy().reshape(-1)
        return out.astype(np.float32), mod, info
    print('DDSP 추론 준비 완료 · 체크포인트 폴더:', CKPT_ROOT)
else:
    print('DDSP 없음 → 사전 렌더 예시 폴더에서 듣습니다:', EXAMPLE_DIR, sorted(os.listdir(EXAMPLE_DIR)) if os.path.isdir(EXAMPLE_DIR) else '(없음)')'''),
("code", r'''#@title 추론: 악기 4종 변환 (악기당 CPU 10–20초, 첫 실행은 체크포인트 다운로드 포함)
results, controls, infos = {}, {}, {}
for name in INSTRUMENTS:
    t0 = time.time()
    if DDSP_OK:
        results[name], controls[name], infos[name] = tone_transfer(name, feats)
        print('%-16s %5.1f s | 옥타브 이동 %+d (입력 평균 MIDI %.1f → 악기 평균 %.1f)%s' % (name, time.time() - t0, infos[name].get('octave_shift', 0), infos[name].get('input_mean_midi', 0), infos[name].get('target_mean_midi', 0), ' | loudness 자동 보정 %+.0f dB' % infos[name]['auto_loudness_gain_db'] if infos[name].get('auto_loudness_gain_db') else ''))
    else:
        p = os.path.join(EXAMPLE_DIR, name + '.wav')
        if os.path.exists(p): results[name], _ = librosa.load(p, sr=TARGET_SR, mono=True); print('%-16s 사전 렌더 예시 로드' % name)
        else: print('%-16s 예시 없음' % name)
if not DDSP_OK:
    ex_in = [p for p in glob.glob(os.path.join(EXAMPLE_DIR, 'input*.wav'))]
    if ex_in: example_input, _ = librosa.load(ex_in[0], sr=TARGET_SR, mono=True); print('예시의 입력:', os.path.basename(ex_in[0]))
    print('내 음원으로 직접 변환하려면 웹 데모를 쓰세요: https://sites.research.google/tonetransfer (브라우저에서 녹음·업로드 → 악기 선택)')'''),
("code", r'''#@title 듣기·보기: 입력 vs 악기별 출력
def norm(y): return y / (np.abs(y).max() + 1e-9) * 0.8
if DDSP_OK:
    fig, ax = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    plot_controls(feats['f0_hz'], feats['loudness_db'], feats['f0_confidence'], ax=ax, title='input f0 (solid) vs. f0 fed to each model (dashed, octave-shifted)')
    for name, c in zip(INSTRUMENTS, ['#83c167', '#f5d547', '#fc6255', '#b19cd9']):
        f = controls[name]['f0_hz'].copy(); f[~(feats['f0_confidence'] > 0.7)] = np.nan; ax[0].plot(np.arange(len(f)) / FRAME_RATE, f, color=c, lw=.8, ls='--', label='%s (%+d oct)' % (name, infos[name].get('octave_shift', 0)))
        ax[1].plot(np.arange(len(f)) / FRAME_RATE, controls[name]['loudness_db'], color=c, lw=.8, ls='--')
    ax[0].legend(loc='best', ncol=4, fontsize=8); plt.tight_layout(); plt.show()
    print('입력'); display(Audio(mono, rate=sr))
else:
    if 'example_input' in dir(): print('예시의 입력 (트럼펫)'); display(Audio(example_input, rate=TARGET_SR))
for name, y in results.items():
    print(name); display(Audio(norm(y), rate=TARGET_SR))'''),
("md", """## 해 보기

1. 아래 슬라이더로 **옥타브**와 **음량**을 바꿔 다시 변환해 보세요. 모델이 보는 것은 두 곡선뿐이므로, 곡선을 바꾸면 출력이 그대로 따라옵니다. (f0를 +1 옥타브 올리면 바이올린은 자연스럽지만 색소폰은? 학습 음역 밖의 f0에서는 어떤 소리가 나나요?)
2. 목소리(마이크 녹음 또는 04의 보컬 스템)를 넣었을 때 **어느 악기가 가장 잘 어울리나요? 왜?** (힌트: 위 그래프에서 입력 평균 음높이와 악기 평균 음높이의 거리, 숨·자음 구간의 confidence, 비브라토의 깊이를 보세요.)
3. 한 번에 한 악기만 다시 돌리므로 CPU에서도 10초 정도면 됩니다."""),
("code", r'''#@title 조정: 옥타브 · 음량 바꿔서 한 악기만 다시 변환
INSTRUMENT = 'violin'    #@param ["violin", "flute", "trumpet", "tenor_saxophone"]
PITCH_SHIFT_OCTAVES = 0  #@param {type:"slider", min:-1, max:1, step:1}
LOUDNESS_SHIFT_DB = 0    #@param {type:"slider", min:-20, max:20, step:2}
if DDSP_OK:
    t0 = time.time(); y_try, ctrl_try, info_try = tone_transfer(INSTRUMENT, feats, pitch_shift=PITCH_SHIFT_OCTAVES, loudness_shift=LOUDNESS_SHIFT_DB)
    print('%s | 옥타브 %+d (자동 %+d + 수동 %+d) | 음량 %+d dB | %.1f s' % (INSTRUMENT, info_try.get('octave_shift', 0) + PITCH_SHIFT_OCTAVES, info_try.get('octave_shift', 0), PITCH_SHIFT_OCTAVES, LOUDNESS_SHIFT_DB, time.time() - t0))
    fig, ax = plt.subplots(2, 1, figsize=(14, 4), sharex=True); plot_controls(ctrl_try['f0_hz'], ctrl_try['loudness_db'], ax=ax, title='controls fed to the model (%s)' % INSTRUMENT, color='#83c167'); plt.tight_layout(); plt.show()
    display(Audio(norm(y_try), rate=TARGET_SR)); results[INSTRUMENT + '_try'] = y_try
else:
    y_try = sine_from_controls(feats['f0_hz'] * 2.0 ** PITCH_SHIFT_OCTAVES, feats['loudness_db'] + LOUDNESS_SHIFT_DB, feats['f0_confidence'])
    print('DDSP 없음 → 바뀐 곡선을 사인파로만 들려줍니다 (옥타브 %+d, 음량 %+d dB). 악기 음색은 웹 데모에서: https://sites.research.google/tonetransfer' % (PITCH_SHIFT_OCTAVES, LOUDNESS_SHIFT_DB))
    display(Audio(y_try, rate=TARGET_SR))'''),
("code", r'''#@title 저장: 결과 wav와 제어 곡선을 outputs/tone_transfer 폴더에
tt_dir = os.path.join(OUT_DIR, 'tone_transfer'); os.makedirs(tt_dir, exist_ok=True)
sf.write(os.path.join(tt_dir, 'input_16k.wav'), mono, TARGET_SR); sf.write(os.path.join(tt_dir, 'sine_resynth.wav'), sine, TARGET_SR)
for name, y in results.items(): sf.write(os.path.join(tt_dir, name + '.wav'), norm(y), TARGET_SR)
json.dump({'fps': FRAME_RATE, 'f0_method': F0_METHOD, 'ddsp': DDSP_OK, 'f0_hz': feats['f0_hz'].tolist(), 'f0_confidence': feats['f0_confidence'].tolist(), 'loudness_db': feats['loudness_db'].tolist()}, open(os.path.join(tt_dir, 'controls.json'), 'w'))
print('저장:', sorted(os.listdir(tt_dir)))'''),
]
build("05_ddsp_tone_transfer.ipynb", cells, "05 DDSP Tone Transfer")
