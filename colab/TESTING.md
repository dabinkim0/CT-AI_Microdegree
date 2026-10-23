# 로컬 실행과 검증

## 로컬 실행 환경

노트북별 패키지가 서로 충돌하므로 conda 환경을 다섯 개로 나눈다. 환경 파일은 `envs/`에 있으며, 패키지 목록은 Colab 설치 셀과 같은 `requirements/` 파일을 사용한다.

| 환경 | 노트북 | 주요 패키지 |
|---|---|---|
| `ctai-analysis` | 00, 01, 04 | torch 2.6.0 (CPU), librosa, pretty_midi, demucs 4.0.1 |
| `ctai-mt3` | 02 | torch 2.6.0 (CPU), transformers 4.45.1, lightning |
| `ctai-allin1` | 03 | torch 2.6.0 (CPU), madmom (GitHub), allin1 1.1.0 (NATTEN 대신 `natten_torch.py`) |
| `ctai-ddsp` | 05 | tensorflow 2.21.0, tf-keras 2.21.0, ddsp 3.7.0, crepe 0.0.16 |
| `ctai-sa3` | 06, 07 | ai-edge-litert, sentencepiece |

모든 환경은 Python 3.12이다. 환경은 `envs/setup.sh`로 만든다. allin1과 ddsp는 pip 옵션이 따로 필요한 패키지가 있어 `conda env create -f`만으로는 설치가 끝나지 않는다.

```bash
bash colab/envs/setup.sh analysis              # ctai-analysis 환경 생성
bash colab/envs/setup.sh analysis ~/envs/ana   # 경로를 지정할 때 (conda create --prefix와 같음)
conda activate ctai-analysis
jupyter lab colab/notebooks
```

- torch는 CPU 빌드이다. NVIDIA GPU에서 쓰려면 yml의 `--extra-index-url` 주소를 `https://download.pytorch.org/whl/cu126`으로 바꾼다.
- 03은 GPU가 없어도 allin1 경로로 구간 라벨까지 낸다. 이웃 어텐션은 `natten_torch.py`(순수 PyTorch)가 맡는다.
- `git`(01·06·07의 코드 받기)과 `ffmpeg`(YouTube 입력)는 conda-forge에서 환경에 함께 깐다. 시스템에 없어도 된다.
- Linux x86_64와 macOS arm64(Apple Silicon)에서 검증했다.

### macOS (Apple Silicon)

conda가 없으면 [Miniforge](https://github.com/conda-forge/miniforge)를 설치한다.

```bash
curl -fsSLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

- 03(`allin1` 환경)은 madmom을 C로 빌드하므로 Xcode 명령줄 도구가 필요하다. `xcode-select --install`로 먼저 설치한다. 없으면 `setup.sh allin1`이 안내 문구를 내고 멈춘다. 나머지 네 환경은 명령줄 도구 없이 만들어진다.
- 04는 Apple Silicon GPU(MPS)를 쓴다. 60초 곡 분리가 CPU 18.6 s → MPS 4.2 s이고, 같은 시드에서 CPU 결과와의 최대 차이는 1e-5 이하이다.
- 「업로드」를 고르면 Colab의 파일 선택 창 대신 파일 경로를 입력받는다. 「마이크 녹음」(05)은 Colab에서만 된다.
- 디스크: 환경 다섯 개 약 7.3 GB, 모델 가중치 약 5.5 GB, pip·conda 캐시 약 1.4 GB.
- 메모리: 노트북 하나의 최대 사용량은 02가 4.6 GB로 가장 크다. 16 GB 맥에서 한 번에 하나씩 실행하면 충분하다.

#### 검증 결과 (macOS)

2026-10-23, Apple M3 Pro 18 GB, macOS 15.2. 새 맥북을 흉내 내려고 Miniforge만 새로 깔고, Homebrew를 PATH에서 빼고, Xcode 명령줄 도구를 가린 채(`DEVELOPER_DIR`을 없는 경로로) 빈 HOME에서 `setup.sh`로 환경을 만들고 설치 셀까지 포함해 nbconvert로 실행했다. 03만 명령줄 도구를 켠 상태에서 만들었다. 노트북들은 동시에 실행했고, 실행 시간에는 가중치를 처음 받는 시간이 들어 있다.

| 환경 | 환경 생성 | 노트북 | 결과 | 실행 시간 | 최대 메모리 |
|---|---|---|---|---|---|
| `ctai-analysis` | 155 s | 00 / 01 / 04 | 통과 | 72 s / 268 s / 75 s | 0.5 / 0.7 / 1.7 GB |
| `ctai-mt3` | 122 s | 02 | 통과 | 254 s | 4.6 GB |
| `ctai-allin1` | 173 s | 03 (madmom 경로) | 통과 | 22 s | 1.3 GB |
| `ctai-ddsp` | 114 s | 05 | 통과 | 173 s | 0.9 GB |
| `ctai-sa3` | 27 s | 06 / 07 | 통과 | 321 s / 93 s | 3.6 / 2.3 GB |

실제 M5 칩과 macOS 26에서는 확인하지 않았다. 팬이 없는 맥북 에어에서는 06·07처럼 오래 도는 노트북이 위 시간보다 느려질 수 있다.

### 검증 결과

2026-10-23, Linux x86_64 CPU 20코어 (`CUDA_VISIBLE_DEVICES=""`). 각 환경을 새로 만든 뒤 `CTAI_SKIP_INSTALL=1`로 `run_local_tests.sh`를 실행했다. 다섯 환경을 동시에 실행했고, 실행 시간은 05를 제외하면 모델 가중치를 이미 받아 둔 상태의 값이다. 실행 전후 `pip freeze`가 같아 노트북 설치 셀이 추가로 설치한 패키지는 없다.

| 환경 | 환경 생성 | 노트북 | 결과 | 실행 시간 |
|---|---|---|---|---|
| `ctai-analysis` | 159 s | 00 / 01 / 04 | 통과 | 14 s / 21 s / 37 s |
| `ctai-mt3` | 198 s | 02 | 통과 | 116 s |
| `ctai-allin1` | 212 s | 03 (madmom 경로) | 통과 | 63 s |
| `ctai-ddsp` | 293 s | 05 | 통과 | 78 s |
| `ctai-sa3` | 57 s | 06 / 07 | 통과 | 54 s / 74 s |

가중치를 처음 받을 때는 01이 약 5분, 02가 약 3분, 06이 약 3분 걸렸다.

## 노트북 고치기

노트북(`.ipynb`)은 `build/`의 스크립트가 만든다. 내용을 고칠 때는 스크립트를 고치고 다시 실행한다 (`nbformat` 필요).

```bash
cd colab/build && for f in gen_0*.py build_nb0*.py; do python $f; done
```

## 검증 스크립트

Colab 없이 CPU로 끝까지 실행해 보려면 `run_local_tests.sh`를 쓴다.

```bash
conda activate ctai-analysis
CTAI_SKIP_INSTALL=1 bash colab/run_local_tests.sh jupyter - 00_getting_started 01_online_amt_piano_transcription 04_demucs_stem_separation
```

다음 환경변수로 경로를 바꿀 수 있다.

| 변수 | 뜻 |
|---|---|
| `CTAI_REPO_DIR` | 이 저장소 위치 (스크립트가 자동 설정) |
| `CTAI_MODELS_DIR` | 모델 폴더 (기본 `~/ctai_class/models`) |
| `CTAI_OUT_DIR` | 결과 폴더 |
| `CTAI_WORK` | 02·03의 임시 작업 폴더 |
| `CTAI_SKIP_INSTALL=1` | 02·03의 `%pip install` 건너뛰기 (환경을 미리 맞춘 경우) |
| `CTAI_FORCE_MADMOM=1` | 03에서 allin1 대신 madmom 대체 경로 강제 |

참고 사항
- 02·03의 손잡이 셀은 `ipywidgets.interact` 대신 `observe`를 쓴다. `interact`는 nbconvert 실행에서 멈춘다.
- 03의 allin1 1.1.0은 NATTEN 0.17의 함수 4개(`natten1dqkrpb`, `natten1dav`, `natten2dqkrpb`, `natten2dav`)를 쓴다. NATTEN 0.17은 Python 3.12·CUDA용 휠만 있어 Colab(Python 3.13)과 맥에서 깔리지 않으므로, 같은 계산을 순수 PyTorch로 옮긴 `natten_torch.py`를 `natten`으로 등록한다. 맥에서 소스로 빌드한 NATTEN 0.17.5와 비교해 함수 단위 67가지 경우에서 최대 오차 1.9e-6, 60초 곡 전체 분석에서 비트·구간이 같고 확률 곡선 차이 1.8e-7이었다.
- 03은 Demucs 분리를 노트북에서 직접 해서 allin1이 찾는 위치에 저장한다. allin1이 부르는 `demucs.separate`는 torchaudio로 wav를 저장하는데, torchaudio 2.9부터는 torchcodec이 필요하다.
- 03은 torch와 numpy를 Colab 기본값 그대로 쓴다(2026-10 기준 Python 3.13, torch 2.11, numpy 2). madmom은 GitHub판을 numpy 2로 빌드한다. 2026-10-24 Colab에서 세션 재시작 없이 설치 약 2분, 60초 곡 분석 T4 12.3 s · CPU 170 s로 allin1 경로(구간 라벨 포함)를 확인했다. 맥(M3 Pro, MPS)에서는 15 s.
- 05는 `TF_USE_LEGACY_KERAS=1`과 `tf-keras`로 ddsp 3.7을 최신 TensorFlow에서 돌린다.
- 06·07은 Stable Audio 3의 CPU 런타임(LiteRT)을 쓴다. 공식 저장소 커밋 `77e0398`에 고정했다.
