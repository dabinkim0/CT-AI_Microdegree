# 노트북 고치기와 검증

노트북(`.ipynb`)은 `build/`의 스크립트가 만든다. 내용을 고칠 때는 스크립트를 고치고 다시 실행한다 (`nbformat` 필요).

```bash
cd colab/build && for f in gen_0*.py build_nb0*.py; do python $f; done
```

Colab 없이 CPU로 검증할 때는 `run_local_tests.sh`를 쓴다. 다음 환경변수로 경로를 바꿀 수 있다.

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
