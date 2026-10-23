from _nb import *

cells = [
("md", """# 08 · Stable Audio 3: 텍스트 → 효과음

앞의 음악 실습(06·07)에서 쓴 Stable Audio 3의 **Small SFX** 모델로 짧은 효과음을 만듭니다. 같은 CPU 런타임(LiteRT)을 사용하며 GPU와 Hugging Face 로그인이 필요 없습니다.

- 입력 → 출력: 음원·소리 이벤트·공간·녹음 특성 → 44.1 kHz 스테레오 효과음
- 가중치: 처음부터 받으면 약 1.19 GB. 앞 실습에서 기본 `medium`을 같은 모델 폴더에 받았다면 약 629 MB, `sm-music`까지 받았다면 약 539 MB가 추가로 필요합니다.
- 가중치 라이선스: Stability AI Community License. 상업적 이용 전 조건을 확인하세요.
- 코드: https://github.com/Stability-AI/stable-audio-3 · 가중치: https://huggingface.co/stabilityai/stable-audio-3-optimized"""),
("code", BOOTSTRAP.replace('KEEP_IN_DRIVE = False',
                            '#@markdown Mac CPU 2코어 실행 실측: 1초 미만.\nKEEP_IN_DRIVE = False')),
("code", r'''#@title 설치와 모델 준비
#@markdown 기본 `sm-sfx`는 효과음용 작은 모델(최대 120초)입니다. 이전 06·07에서 `KEEP_IN_DRIVE`를 켰다면 여기서도 켜 같은 `models` 폴더를 사용하세요.
#@markdown Mac CPU 2코어 실측: 첫 다운로드 포함 약 235초, 로컬 캐시 재사용 약 1초.
MODEL = 'sm-sfx'  #@param ["sm-sfx", "sm-music", "medium"]
%pip install -q -r {REQ}/sa3.txt
import sa3
sa3.prepare(MODELS_DIR, OUT_DIR, dit=MODEL, decoder='same-l' if MODEL == 'medium' else 'same-s', threads=2)
print('설치 OK')'''),
("md", """## 프롬프트의 네 요소

**음원**은 무엇이 소리 나는지, **소리 이벤트**는 무슨 일이 일어나는지, **공간**은 어디서 일어나는지, **녹음 특성**은 어떤 거리·질감으로 담을지 적습니다. 아래 문장은 바꿔 쓰는 예시입니다. 한 번에 모두 바꾸기보다 요소 하나씩 바꿔 결과를 비교해 보세요."""),
("code", r'''#@title 프롬프트와 설정: 네 요소를 따로 적기
#@markdown Mac CPU 2코어 실행 실측: 1초 미만.
SOURCE = 'A metal key ring'  #@param {type:"string"}
EVENT = 'several keys jingle twice as someone lifts the ring'  #@param {type:"string"}
SPACE = 'a small carpeted entryway'  #@param {type:"string"}
RECORDING = 'captured by a portable recorder near the keys, with very little room echo'  #@param {type:"string"}
PROMPT = f'{SOURCE}; {EVENT}; {SPACE}; {RECORDING}.'
SECONDS = 4  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
STEPS = 8  #@param {type:"integer"}
print('PROMPT:', PROMPT)
print(f'길이 {SECONDS}초 · seed {SEED} · steps {STEPS}')'''),
("md", """## 같은 프롬프트·시드에서 CFG 비교

CFG 1과 4를 차례로 만듭니다. CFG는 프롬프트 반영 강도이며 큰 값이 언제나 더 좋은 소리를 뜻하지는 않습니다. 비교 그림의 음량(RMS·Peak)과 청취 결과를 함께 살펴보세요. 이 셀은 **Mac CPU 2코어 실측 약 17~27초**입니다."""),
("code", r'''#@title 생성: 같은 프롬프트·seed로 CFG 1과 4
#@markdown Mac CPU 2코어 실측: 약 17~27초(4초 WAV 두 개).
t0 = time.time()
comparison_dir, cfg_clips = sa3.generate_cfg_sweep(PROMPT, cfg_values=(1.0, 4.0), seconds=SECONDS,
                                                   seed=SEED, steps=STEPS)
print('CFG 두 개 생성: %.1f s · %s' % (time.time() - t0, comparison_dir))'''),
("code", r'''#@title 비교: CFG 1과 4의 스펙트로그램·플레이어
#@markdown Mac CPU 2코어 실측: 약 2초(렌더링 첫 실행은 38초).
sa3.compare_audio(cfg_clips, title=f'SFX CFG comparison | seed {SEED}', output_dir=comparison_dir,
                  filename='cfg_comparison')'''),
("md", """## Negative prompt 비교

같은 CFG 4에서 negative prompt가 **없는 결과**와 **있는 결과**를 비교합니다. `music, speech, reverb`는 억제하려는 소리의 예시입니다. **CFG 1에서는 negative prompt를 쓸 수 없습니다.** 이 셀은 **Mac CPU 2코어 실측 약 10~22초**입니다."""),
("code", r'''#@title 생성: negative prompt 없음과 있음 (CFG 4)
#@markdown Mac CPU 2코어 실측: 약 10~22초(새 4초 WAV 하나와 비교 그림).
from pathlib import Path
NEGATIVE_PROMPT = 'music, speech, long reverb'  #@param {type:"string"}
NEGATIVE_CFG = 4.0
t0 = time.time()
negative_clips = [('없음', cfg_clips[1][1])]
negative_with = sa3.generate(PROMPT, seconds=SECONDS, seed=SEED, steps=STEPS,
                             cfg=NEGATIVE_CFG, negative_prompt=NEGATIVE_PROMPT)
negative_clips.append((f'있음: {NEGATIVE_PROMPT}', negative_with))
print('negative prompt 추가 생성: %.1f s' % (time.time() - t0))
sa3.compare_audio(negative_clips, title='SFX negative prompt comparison',
                  output_dir=Path(negative_with).parent, filename='negative_comparison')'''),
("md", """## 길이 조정

짧은 **one-shot**과 긴 **ambience**의 `SECONDS`를 따로 정합니다. 아래의 긴 장면 문장도 바꿔 쓰는 예시입니다. WAV 파일의 실제 길이를 출력으로 확인하세요. 기본 2초·8초 생성은 **Mac CPU 2코어 실측 약 13~29초**입니다. Small 모델의 최대 길이는 120초입니다."""),
("code", r'''#@title 길이 조정: 짧은 one-shot과 긴 ambience
#@markdown Mac CPU 2코어 실측: 약 13~29초(2초·8초 WAV).
from pathlib import Path
import soundfile as sf
SHORT_SECONDS = 2  #@param {type:"integer"}
LONG_SECONDS = 8  #@param {type:"integer"}
AMBIENCE_PROMPT = 'An old wooden gate; loose hinges creak intermittently in a steady breeze; an open garden at dusk; recorded from a few steps away with soft outdoor air.'  #@param {type:"string"}
t0 = time.time()
length_clips = []
for label, prompt, seconds in [('one-shot', PROMPT, SHORT_SECONDS),
                               ('ambience', AMBIENCE_PROMPT, LONG_SECONDS)]:
    wav = sa3.generate(prompt, seconds=seconds, seed=SEED, steps=STEPS, cfg=1.0)
    actual = sf.info(str(wav)).duration
    print(f'{label}: 요청 {seconds}초 → 파일 {actual:.3f}초 · {wav}')
    length_clips.append((label, wav))
print('길이 두 개 생성: %.1f s' % (time.time() - t0))
for label, wav in length_clips:
    print(label)
    _ = sa3.show_audio(wav)'''),
("md", """## 요소 하나씩 바꾸기

기준 문장과 시드·길이·CFG를 유지하고 **소리 이벤트**, **공간**, **녹음 특성**을 차례로 하나씩 바꿉니다. 각 출력에 바뀐 요소를 표시합니다. 음원도 바꿔 보고 싶다면 `SOURCE`만 고쳐 같은 방식으로 한 번 더 실행해 보세요. 세 변형은 **Mac CPU 2코어 실측 약 20~23초**입니다."""),
("code", r'''#@title 생성: 요소 하나만 바꾼 세 변형
#@markdown Mac CPU 2코어 실측: 약 20~23초(4초 WAV 세 개와 비교 그림).
base = dict(SOURCE=SOURCE, EVENT=EVENT, SPACE=SPACE, RECORDING=RECORDING)
changes = [
    ('소리 이벤트', 'EVENT', 'a single abrupt shake makes the keys rattle'),
    ('공간', 'SPACE', 'a narrow stone stairwell'),
    ('녹음 특성', 'RECORDING', 'recorded several meters away with a noticeable airy background'),
]
variant_clips = [('기준', cfg_clips[0][1])]
t0 = time.time()
for label, key, value in changes:
    parts = dict(base)
    parts[key] = value
    prompt = '; '.join(parts[k] for k in ('SOURCE', 'EVENT', 'SPACE', 'RECORDING')) + '.'
    print(f'바뀐 요소: {label} ({key}) → {value}')
    wav = sa3.generate(prompt, seconds=SECONDS, seed=SEED, steps=STEPS, cfg=1.0)
    variant_clips.append((label, wav))
print('변형 세 개 생성: %.1f s' % (time.time() - t0))
sa3.compare_audio(variant_clips, title='SFX one-element variations',
                  output_dir=Path(variant_clips[-1][1]).parent, filename='element_comparison')'''),
("code", r'''#@title 저장: 이번 실습 결과를 zip으로 내려받기
#@markdown Mac CPU 2코어 실측: 약 1초(결과 8개 ZIP).
import shutil
job_dirs = {Path(comparison_dir), Path(negative_with).parent}
job_dirs.update(Path(wav).parent for _, wav in length_clips)
job_dirs.update(Path(wav).parent for _, wav in variant_clips[1:])
bundle_dir = Path(OUT_DIR) / ('stable_audio_3_sfx_08_' + Path(comparison_dir).name)
bundle_dir.mkdir(parents=True, exist_ok=True)
for job_dir in sorted(job_dirs):
    shutil.copytree(job_dir, bundle_dir / job_dir.name, dirs_exist_ok=True)
zip_path = shutil.make_archive(str(bundle_dir), 'zip', str(bundle_dir))
print('저장:', zip_path, '(각 폴더의 run.json·WAV와 비교 HTML·PNG·JSON 포함)')
if IN_COLAB and not OUT_DIR.startswith('/content/drive'):
    from google.colab import files; files.download(zip_path)'''),
]

if __name__ == '__main__':
    build('08_stable_audio_3_sfx.ipynb', cells, '08 Stable Audio 3 SFX', gpu=False)
