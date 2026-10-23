# 실습 노트북

KAIST CT×AI 콘텐츠 마이크로디그리 「뉴럴 네트워크 활용하기」의 실습 노트북이다. 표의 링크로 Google Colab에서 연다.

| # | 노트북 | 입력 → 출력 | 런타임 |
|---|---|---|---|
| 00 | [시작하기](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/00_getting_started.ipynb) | 오디오 보기·듣기 | CPU |
| 01 | [Online AMT](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/01_online_amt_piano_transcription.ipynb) | 피아노 오디오 → MIDI (실시간) | CPU |
| 02 | [YourMT3+](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/02_YourMT3plus_multi_instrument_transcription.ipynb) | 음악 오디오 → 멀티악기 MIDI | T4 GPU 권장 |
| 03 | [allin1](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/03_allin1_beat_structure.ipynb) | 음악 오디오 → BPM · 비트 · 구간 | T4 GPU 권장 (CPU도 가능) |
| 04 | [Demucs](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/04_demucs_stem_separation.ipynb) | 음악 오디오 → 6 스템 | T4 GPU 권장 |
| 05 | [Tone Transfer (DDSP)](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/05_ddsp_tone_transfer.ipynb) | 단선율 오디오 → 악기 소리 | CPU 또는 T4 |
| 06 | [Stable Audio 3 생성](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/06_stable_audio_3_generate.ipynb) | 텍스트 → 음악 (CFG 비교) | CPU |
| 07 | [Stable Audio 3 인페인팅](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/07_stable_audio_3_inpaint.ipynb) | 음악 + 텍스트 → 구간만 바뀐 음악 | CPU |
| 08 | [Stable Audio 3 효과음](https://colab.research.google.com/github/dabinkim0/CT-AI_Microdegree/blob/main/colab/notebooks/08_stable_audio_3_sfx.ipynb) | 텍스트 → 효과음 | CPU |

## 실행 방법

1. 표의 링크로 노트북을 연다. 런타임이 GPU인 노트북은 런타임 → 런타임 유형 변경에서 T4 GPU를 선택한다.
2. 셀을 위에서부터 차례로 실행한다. 첫 셀 「0. 준비」가 이 저장소를 내려받고 모델 폴더를 만든다.
3. 필요한 패키지와 모델 가중치는 각 노트북의 설치 셀이 처리한다.

## 저장 위치

모델 가중치는 `models`, 결과 파일은 `outputs`에 저장한다. 기본 위치는 `/content`이며 런타임이 종료되면 삭제된다. 「0. 준비」 셀에서 `KEEP_IN_DRIVE`를 선택하면 Google Drive의 `CT-AI_Microdegree/` 아래에 저장되어 다음 실행 때 다시 내려받지 않는다. 04에서 분리한 보컬을 05에서 쓰려면 두 노트북 모두 이 옵션을 선택한다.

## 모델 가중치

| 노트북 | 하위 폴더 | 크기 | 출처 |
|---|---|---|---|
| 01 | `online_amt/` 코드와 체크포인트 | 약 180 MB | [jdasam/online_amt](https://github.com/jdasam/online_amt) (MIT) |
| 02 | `YourMT3/` 코드와 체크포인트 | 약 540 MB | [HF Space mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3) |
| 03 | `huggingface/` allin1·Demucs | 약 90 MB | [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one) |
| 04 | `torch/` htdemucs_6s | 약 50 MB | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) |
| 05 | `ddsp/` 악기 4종 | 약 230 MB | Google Magenta DDSP |
| 06·07 | `stable-audio-3/` 코드와 CPU 캐시, `huggingface/` 가중치 | 약 3.2 GB + 캐시 2.3 GB | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) |
| 08 | `huggingface/` Small SFX 가중치 | 약 1.19 GB | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) |

Stable Audio 3 가중치에는 Stability AI Community License가 적용된다. 상업적으로 이용하려면 라이선스 조건을 확인한다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `notebooks/` | 노트북 00~08 |
| `audio/` | 예시 음원 4개와 출처 (`LICENSES.md`, 모두 CC 라이선스) |
| `assets/` | Online AMT 실행 코드, DDSP 사전 렌더링 예시 |
| `ctai.py`, `sa3.py` | 「0. 준비」 셀과 내려받기 함수, Stable Audio 3 실행 함수 |
| `build/` | 노트북 생성 스크립트 |
| `envs/` | 로컬 실행용 conda 환경 파일 |

## 입력 음원

예시 음원 외에 각 노트북의 오디오 준비 셀에서 파일 업로드 또는 YouTube 주소를 입력으로 쓸 수 있다. 저작권이 있는 음원은 수업 내 학습 목적으로만 사용하고 외부에 배포하지 않는다.

## 로컬 실행

Colab 밖에서 실행할 때의 conda 환경 구성과 검증 방법은 [TESTING.md](TESTING.md)에 있다.
