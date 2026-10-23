# 로컬 실행과 검증

## 로컬 실행 환경

노트북별 패키지가 서로 충돌하므로 conda 환경을 다섯 개로 나눈다. 환경 파일은 `envs/`에 있으며, 패키지 목록은 Colab 설치 셀과 같은 `requirements/` 파일을 사용한다.

| 환경 | 노트북 | 주요 패키지 |
|---|---|---|
| `ctai-analysis` | 00, 01, 04 | torch 2.6.0 (CPU), librosa, pretty_midi, demucs 4.0.1 |
| `ctai-mt3` | 02 | torch 2.6.0 (CPU), transformers 4.45.1, lightning |
| `ctai-allin1` | 03 | torch 2.6.0 (CPU), numpy 1.26.4, madmom (GitHub), allin1 1.1.0 |
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
- 03은 GPU가 없으면 NATTEN 없이 madmom으로 비트·다운비트·BPM만 출력한다.
- Linux x86_64에서만 검증했다.

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
- 03의 allin1은 NATTEN 휠이 필요하고, 휠은 CUDA용만 있다. GPU가 없으면 madmom으로 비트·다운비트만 낸다.
- 05는 `TF_USE_LEGACY_KERAS=1`과 `tf-keras`로 ddsp 3.7을 최신 TensorFlow에서 돌린다.
- 06·07은 Stable Audio 3의 CPU 런타임(LiteRT)을 쓴다. 공식 저장소 커밋 `77e0398`에 고정했다.
