#!/bin/bash
# 노트북을 Colab 밖에서 돌릴 conda 환경을 만든다.
# 사용법: bash colab/envs/setup.sh <analysis|mt3|allin1|ddsp|sa3> [설치 경로]
# 설치 경로를 주지 않으면 ctai-<이름> 환경을 만든다.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
NAME=$1
if [ ! -f "$DIR/$NAME.yml" ]; then
  echo "사용법: bash $0 <analysis|mt3|allin1|ddsp|sa3> [설치 경로]"; exit 1
fi
if [ -n "$2" ]; then WHERE=(-p "$2"); else WHERE=(-n "ctai-$NAME"); fi
if [ "$NAME" = allin1 ] && [ "$(uname)" = Darwin ] && ! xcrun --find clang >/dev/null 2>&1; then
  echo "madmom 빌드에 C 컴파일러가 필요합니다. 'xcode-select --install'로 Xcode 명령줄 도구를 설치한 뒤 다시 실행하세요."; exit 1
fi
conda env create "${WHERE[@]}" -f "$DIR/$NAME.yml"
PIP=(conda run "${WHERE[@]}" --no-capture-output python -m pip install)
case $NAME in
  allin1)  # madmom은 numpy·cython이 깔린 뒤 빌드 격리 없이 설치해야 한다
    "${PIP[@]}" --no-build-isolation -r "$DIR/../requirements/03_madmom.txt"
    "${PIP[@]}" -r "$DIR/../requirements/03_allin1.txt" ;;
  ddsp)    # crepe는 setup.py가 pkg_resources를 쓰고, ddsp 3.7은 메타데이터가 TF<=2.11을 요구한다
    "${PIP[@]}" --no-build-isolation crepe==0.0.16
    "${PIP[@]}" --no-deps ddsp==3.7.0 ;;
esac
echo "완료: conda activate ${WHERE[1]}"
