"""수업 노트북 공통 도우미: 경로 설정과 모델 파일 내려받기.

모든 노트북의 첫 셀(0. 준비)이 이 모듈을 불러 `setup()`을 실행한다.
모델 가중치는 전부 MODELS_DIR 한 폴더 아래에 저장된다.
"""
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_URL = 'https://github.com/dabinkim0/CT-AI_Microdegree'


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup(keep_in_drive=False):
    """경로를 정하고 Hugging Face·PyTorch 캐시를 MODELS_DIR 안으로 돌린다.

    반환: dict(REPO_DIR, AUDIO_DIR, ASSET_DIR, MODELS_DIR, CODE_DIR, OUT_DIR, IN_COLAB)
    환경변수 CTAI_MODELS_DIR, CTAI_OUT_DIR가 있으면 그 경로를 쓴다 (서버 테스트용).
    """
    colab = in_colab()
    repo = Path(__file__).resolve().parent.parent
    if keep_in_drive and colab:
        from google.colab import drive
        drive.mount('/content/drive')
        base = Path('/content/drive/MyDrive/CT-AI_Microdegree')
    elif colab:
        base = Path('/content')
    else:
        base = Path.home() / 'ctai_class'
    models = Path(os.environ.get('CTAI_MODELS_DIR', base / 'models'))
    out = Path(os.environ.get('CTAI_OUT_DIR', base / 'outputs'))
    models.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # 라이브러리가 알아서 받는 가중치(YourMT3+, allin1, Demucs, Stable Audio 3)도 같은 폴더로
    os.environ['HF_HOME'] = str(models / 'huggingface')
    os.environ['TORCH_HOME'] = str(models / 'torch')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    # 코드 저장소(git)는 Drive에 두면 느리고 링크가 깨지므로 Drive 모드에서는 /content/code에 둔다
    code = Path('/content/code') if str(models).startswith('/content/drive') else models
    code.mkdir(parents=True, exist_ok=True)
    paths = dict(REPO_DIR=str(repo), AUDIO_DIR=str(repo / 'colab' / 'audio'),
                 ASSET_DIR=str(repo / 'colab' / 'assets'), MODELS_DIR=str(models), CODE_DIR=str(code),
                 OUT_DIR=str(out), IN_COLAB=colab)
    print('모델 폴더:', models)
    print('결과 폴더:', out)
    return paths


def _human(n):
    return '%.0f MB' % (n / 1e6) if n >= 1e6 else '%.0f KB' % (n / 1e3)


def fetch(urls, dest, size=None):
    """urls(문자열 또는 목록)에서 dest로 내려받는다. 이미 있으면 건너뛴다.

    size를 주면 크기가 맞을 때만 '이미 있음'으로 본다.
    'gdrive:<id>' 형식은 gdown으로 Google Drive에서 받는다.
    """
    dest = Path(dest)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print('이미 있음:', dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    urls = [urls] if isinstance(urls, str) else list(urls)
    last = None
    for url in urls:
        try:
            t0 = time.time()
            if url.startswith('gdrive:'):
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
                import gdown
                gdown.download(id=url.split(':', 1)[1], output=str(tmp), quiet=False)
            else:
                print('내려받는 중:', dest.name)
                req = urllib.request.Request(url, headers={'User-Agent': 'ctai-class'})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
                    total = int(r.headers.get('Content-Length') or 0)
                    done, step = 0, 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total and done * 10 // total > step:
                            step = done * 10 // total
                            print('  %3d%%  %s / %s' % (step * 10, _human(done), _human(total)), flush=True)
            if size is not None and tmp.stat().st_size != size:
                raise IOError('크기가 다릅니다: %d (예상 %d)' % (tmp.stat().st_size, size))
            tmp.replace(dest)
            print('완료: %s (%s, %.0f s)' % (dest.name, _human(dest.stat().st_size), time.time() - t0))
            return dest
        except Exception as e:  # 다음 주소로 재시도
            last = e
            print('  실패:', url[:80], '→', repr(e)[:120])
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError('내려받기에 실패했습니다: %s (%r)' % (dest.name, last))


def git_checkout(url, dest, commit=None, sparse=None):
    """코드 저장소를 dest에 받는다. commit을 주면 그 커밋으로 고정한다."""
    dest = Path(dest)
    if (dest / '.git').exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    run = lambda *a: subprocess.run(['git', '-C', str(dest), *a], check=True, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dest.mkdir(parents=True)
    run('init', '-q')
    run('remote', 'add', 'origin', url)
    if sparse:
        run('sparse-checkout', 'set', '--no-cone', *sparse)
    run('fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', commit or 'HEAD')
    run('checkout', '-q', 'FETCH_HEAD')
    print('코드 준비:', dest)
    return dest
