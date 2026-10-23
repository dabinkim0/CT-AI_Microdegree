from _nb import *
cells = [
("md", """# 00 · 시작하기: 소리를 숫자로 보기

이 수업의 모델은 모두 **오디오를 입력으로 받아 기호(MIDI, 비트, 구간) 또는 다른 오디오를 출력**합니다. 모델을 돌리기 전에, 오디오가 컴퓨터 안에서 어떤 모양인지 먼저 봅니다.

- 소요 시간: 5분 · GPU 불필요
- 이 노트북에서 익히는 것: 준비 셀, 예시 음원·업로드·YouTube 중 고르기, 파형과 스펙트로그램, 재생 위젯
- 이후 노트북(01~07)은 모두 같은 "0. 준비" 셀로 시작합니다. 모델 가중치는 각 노트북이 처음 실행될 때 `models` 폴더에 내려받습니다."""),
("code", BOOTSTRAP),
("code", AUDIO_PICK.format(EXAMPLES="{'밴드 (Let\\'s Go Fishin\\', 보컬+밴드)': 'band_lets_go_fishin_60s.wav', '재즈 (Vibe Ace)': 'jazz_vibe_ace_60s.wav', '피아노 (Scarlatti K.455)': 'piano_scarlatti_k455_30s.wav', '트럼펫 솔로 (5초)': 'trumpet_solo_5s.wav'}",
    DEFAULT_EXAMPLE="밴드 (Let's Go Fishin', 보컬+밴드)", EXAMPLE_PARAM='["밴드 (Let\'s Go Fishin\', 보컬+밴드)", "재즈 (Vibe Ace)", "피아노 (Scarlatti K.455)", "트럼펫 솔로 (5초)"]', MAX_SECONDS=30, TARGET_SR=22050, MONO=True)),
("md", """## 1초에 22,050개의 숫자

위 파형은 1초에 22,050개(샘플링 레이트)의 숫자를 선으로 이은 것입니다. 모델은 이 숫자열을 직접 보거나, 아래처럼 **시간-주파수 그림(스펙트로그램)**으로 바꿔서 봅니다."""),
("code", r'''#@title 스펙트로그램: 모델이 "보는" 그림
import librosa.display
S = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)), ref=np.max)
M = librosa.power_to_db(librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=128), ref=np.max)
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=ax[0], cmap='magma'); ax[0].set_title('STFT (log-frequency)')
librosa.display.specshow(M, sr=sr, x_axis='time', y_axis='mel', ax=ax[1], cmap='magma'); ax[1].set_title('mel-spectrogram (피아노 채보 모델의 입력)')
plt.tight_layout(); plt.show()
print('STFT 행렬 크기:', S.shape, '(주파수 bin × 시간 프레임)  | mel:', M.shape)'''),
("md", """## 입력 → 출력으로 모델 나누기 (강의 슬라이드 THU 16)

| 유형 | 입력 → 출력 | 노트북 |
|---|---|---|
| A. 분석 | 피아노 오디오 → MIDI | 01 Online AMT |
| A. 분석 | 음악 오디오 → 멀티악기 MIDI | 02 YourMT3+ |
| A. 분석 | 음악 오디오 → BPM · 비트 · 구간 | 03 Allin1 |
| B. 편집 | 음악 오디오 → 6 스템 | 04 Demucs |
| B. 편집 | 단선율 오디오 → 단선율 오디오 | 05 Tone Transfer (DDSP) |
| C. 생성 | 텍스트 → 음악 오디오 | 06 Stable Audio 3 생성 |
| C. 생성 | 음악 오디오 + 텍스트 → 음악 오디오 | 07 Stable Audio 3 인페인팅 |

**해 보기**: `SOURCE`를 바꿔 자기 음원을 넣어 보세요. 스펙트로그램에서 드럼(세로 줄)과 보컬·악기(가로 줄)가 어떻게 다르게 보이나요?"""),
("code", r'''#@title 저장: 다듬은 음원을 outputs 폴더에 저장 (다른 노트북에서 재사용)
out = os.path.join(OUT_DIR, 'my_input.wav'); sf.write(out, mono, sr); print('저장:', out)'''),
]
build("00_getting_started.ipynb", cells, "00 Getting Started")
