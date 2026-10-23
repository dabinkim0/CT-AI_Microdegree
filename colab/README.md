# 실습 노트북 (Colab)

KAIST CT×AI 콘텐츠 마이크로디그리 "뉴럴 네트워크 활용하기" 실습 자료. 아래 링크를 누르면 Colab에서 바로 열린다.

| # | 노트북 | 입력 → 출력 | 런타임 |
|---|---|---|---|
| 00 | [시작하기](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/00_getting_started.ipynb) | 오디오 보기·듣기 | CPU |
| 01 | [Online AMT](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/01_online_amt_piano_transcription.ipynb) | 피아노 오디오 → MIDI (실시간) | CPU |
| 02 | [YourMT3+](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/02_YourMT3plus_multi_instrument_transcription.ipynb) | 음악 오디오 → 멀티악기 MIDI | T4 GPU 권장 |
| 03 | [allin1](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/03_allin1_beat_structure.ipynb) | 음악 오디오 → BPM · 비트 · 구간 | T4 GPU 필요 (없으면 비트만) |
| 04 | [Demucs](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/04_demucs_stem_separation.ipynb) | 음악 오디오 → 6 스템 | T4 GPU 권장 |
| 05 | [Tone Transfer (DDSP)](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/05_ddsp_tone_transfer.ipynb) | 단선율 오디오 → 악기 소리 | CPU 또는 T4 |
| 06 | [Stable Audio 3 생성](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/06_stable_audio_3_generate.ipynb) | 텍스트 → 음악 (CFG 비교) | CPU |
| 07 | [Stable Audio 3 인페인팅](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/07_stable_audio_3_inpaint.ipynb) | 음악 + 텍스트 → 구간만 바뀐 음악 | CPU |

## 사용법

1. 링크로 노트북을 연다. GPU가 필요한 노트북은 런타임 → 런타임 유형 변경 → **T4 GPU**.
2. 위에서부터 셀을 차례로 실행한다. 첫 셀 **0. 준비**가 이 저장소를 받고 모델 폴더를 만든다.
3. 모델 가중치는 각 노트북이 처음 실행될 때 `models` 폴더 하나에 내려받는다. 따로 설치할 것은 없다.

`models`의 기본 위치는 `/content/models`이고 런타임을 끄면 지워진다. 0. 준비 셀에서 `KEEP_IN_DRIVE`를 켜면 Google Drive의 `CT-AI_Microdegree/` 아래에 `models`와 `outputs`를 두어 다음에 다시 받지 않는다. 04의 보컬 스템을 05에서 쓰려면 두 노트북 모두 이 옵션을 켠다.

## models 폴더에 받는 것

| 노트북 | 하위 폴더 | 크기 | 출처 |
|---|---|---|---|
| 01 | `online_amt/` 코드와 체크포인트 | 약 180 MB | [jdasam/online_amt](https://github.com/jdasam/online_amt) (MIT) |
| 02 | `YourMT3/` 코드와 체크포인트 | 약 540 MB | [HF Space mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3) |
| 03 | `huggingface/` allin1·Demucs, `wheels/` NATTEN (GPU일 때) | 약 90 MB + 475 MB | [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one), [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) |
| 04 | `torch/` htdemucs_6s | 약 50 MB | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) |
| 05 | `ddsp/` 악기 4종 | 약 230 MB | Google Magenta DDSP |
| 06·07 | `stable-audio-3/` 코드와 CPU 캐시, `huggingface/` 가중치 | 약 3.2 GB + 캐시 2.3 GB | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) |

Stable Audio 3 가중치는 Stability AI Community License를 따른다. 상업 서비스에 쓸 때는 라이선스를 확인한다.

## 폴더 구성

- `notebooks/` 노트북 00~07
- `audio/` 예시 음원 4개와 출처 (`LICENSES.md`, 모두 CC 라이선스)
- `assets/` online AMT 실행 코드, DDSP 미리 렌더한 예시 (설치가 안 될 때 듣기용)
- `ctai.py` 0. 준비 셀과 내려받기 도우미, `sa3.py` Stable Audio 3 도우미
- `build/` 노트북 생성 스크립트, `TESTING.md` 고치기와 검증 방법

## 자기 음원 쓰기

각 노트북의 오디오 준비 셀에서 업로드나 YouTube를 고른다. 저작권이 있는 음원은 수업 안에서 학습 목적으로만 쓰고 외부에 배포하지 않는다.

## 문제가 생기면

- 설치 셀 오류: 런타임 → 런타임 다시 시작 후 처음부터 다시 실행.
- 03에서 NATTEN 설치가 안 되면 madmom 대체 경로(비트·다운비트·BPM만)로 자동 전환된다.
- 05에서 ddsp 설치가 안 되면 f0·음량 추출과 미리 렌더한 예시로 진행하고, 웹 데모 https://sites.research.google/tonetransfer 를 쓴다.
- 06·07이 느리면 설치 셀에서 `MODEL`을 `sm-music`으로 바꾸거나 `SECONDS`를 줄인다.
