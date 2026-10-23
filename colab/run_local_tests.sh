#!/bin/bash
# 노트북을 Colab 없이 CPU로 끝까지 실행해 검증한다.
# 사용법: bash colab/run_local_tests.sh <jupyter 실행 파일> <커널 이름 또는 -> <노트북 이름>...
# 예: bash colab/run_local_tests.sh ~/venvs/sa3/bin/jupyter - 06_stable_audio_3_generate
REPO=$(cd "$(dirname "$0")/.." && pwd)
JUP=$1; KERN=$2; shift 2
export CTAI_REPO_DIR=$REPO CUDA_VISIBLE_DEVICES=""
export CTAI_MODELS_DIR=${CTAI_MODELS_DIR:-$HOME/ctai_class/models} CTAI_OUT_DIR=${CTAI_OUT_DIR:-/tmp/ctai_out}
KARGS=""; [ "$KERN" != "-" ] && KARGS="--ExecutePreprocessor.kernel_name=$KERN"
for n in "$@"; do
  "$JUP" nbconvert --to notebook --execute $KARGS --ExecutePreprocessor.timeout=2400 \
    --output-dir /tmp/ctai_exec --output "${n}_out" "$REPO/colab/notebooks/$n.ipynb" 2>&1 | grep -vE '^\[NbConvertApp\]' | tail -3
done
