from _nb import *
from gen_06 import SA3_INTRO, SA3_INSTALL

cells = [
("md", """# 07 · Stable Audio 3: 구간 인페인팅 (음악 + 텍스트 → 음악)

""" + SA3_INTRO + """

**인페인팅**은 곡의 한 구간만 노이즈에서 다시 만들고, 구간 밖은 문맥으로 씁니다. 기본 예제는 잔잔한 피아노 24초 중 **6~18초를 웅장한 오케스트라**로 바꿉니다. 원본 파일은 바꾸지 않습니다.

- 입력 → 출력: 음악 오디오 + 텍스트 + 구간 → 그 구간만 바뀐 음악 오디오
- 소요 시간: 설치 1분 + 가중치 내려받기 2~3분(처음 한 번, 06과 같은 파일) + CPU 2코어 기준 피아노 생성 약 40초, 인페인팅 약 65초
- GPU 불필요
- 논문: Evans et al., "Stable Audio 3," 2026 (arXiv:2605.17991) · 코드: https://github.com/Stability-AI/stable-audio-3"""),
("code", BOOTSTRAP),
("code", SA3_INSTALL),
("code", r'''#@title 원본과 목표 스타일
SOURCE = '피아노 생성 (24초)'  #@param ["피아노 생성 (24초)", "예시 (재즈 Vibe Ace, 앞 24초)", "업로드"]
START_SECONDS = 6.0  #@param {type:"number"}
END_SECONDS = 18.0   #@param {type:"number"}
TARGET_STYLE = 'orchestral_ensemble'  #@param ["orchestral_ensemble", "drum_and_bass", "heavy_metal"]
SOURCE_PROMPT = ("Instrumental solo felt piano, slow sparse delicate notes, peaceful intimate acoustic ballad, "
                 "gentle dynamics, no drums, no percussion, no synthesizers, no vocals.")
STYLE_PROMPTS = {
    "orchestral_ensemble": "Instrumental cinematic orchestral ensemble, 100 BPM, full acoustic orchestra with lush layered violin and viola sections, rhythmic cellos and double basses, soaring French horns and brass, lyrical woodwinds, timpani rolls and cymbal swells, bold sweeping melody, powerful dramatic crescendo, rich concert hall acoustics, full ensemble rather than solo piano, no synthesizers, no vocals.",
    "drum_and_bass": "Instrumental aggressive drum and bass, 174 BPM, rapid chopped breakbeats, heavy distorted reese bass, sharp snares, bright high-speed hi-hats, intense energetic electronic drop, no vocals.",
    "heavy_metal": "Instrumental heavy metal, fast double-kick drums, heavily distorted electric guitar power-chord riffs, aggressive bass guitar, crashing cymbals, loud energetic full band, no vocals.",
}
EDIT_PROMPT = STYLE_PROMPTS[TARGET_STYLE]
SEED = 73
STEPS = 8
CFG = 4.0   # 원곡 성격을 밀어내려고 negative prompt를 쓰므로 1보다 커야 함
NEGATIVE_PROMPT = "gentle solo piano, quiet acoustic ballad, ambient pads, soft sparse music, vocals, speech"
CROSSFADE_SECONDS = 0.10

if SOURCE.startswith('피아노'):
    source = sa3.generate(SOURCE_PROMPT, seconds=24, seed=72)
elif SOURCE.startswith('예시'):
    source = os.path.join(AUDIO_DIR, 'jazz_vibe_ace_60s.wav')
else:
    source = str(ctai.upload())   # Colab: 파일 선택 창 / 로컬: 파일 경로 입력
MAX_INPUT_SECONDS = 24   # CPU 시간을 아끼려고 앞 24초만 씀
print('원본:', source)
sa3.plot_audio(source, region=(START_SECONDS, END_SECONDS))
_ = sa3.show_audio(source)'''),
("md", """## 구간 재생성

`inpaint`는 한 번의 호출로 ① 입력을 44.1 kHz 스테레오 PCM16 기준 파일로 바꾸고 ② 모델로 구간을 다시 만들고 ③ 그 구간만 기준 파일에 붙입니다(경계 안쪽 0.1초 크로스페이드). 공식 가이드의 경험칙: 구간이 몇 초뿐이면 문맥이 강해서 원곡과 거의 같게 나옵니다. 처음엔 넓게 잡고 점점 줄여 보세요."""),
("code", r'''#@title 인페인팅 실행
t0 = time.time()
edited = sa3.inpaint(source, EDIT_PROMPT, start=START_SECONDS, end=END_SECONDS, seed=SEED, steps=STEPS, cfg=CFG,
                     negative_prompt=NEGATIVE_PROMPT, crossfade=CROSSFADE_SECONDS, max_seconds=MAX_INPUT_SECONDS)
print('%.0f s · 최종 파일: %s' % (time.time() - t0, edited))'''),
("code", r'''#@title 전체 곡 비교: 청록 점선이 편집 경계
from pathlib import Path
report = json.loads((Path(edited).parent / "edit_report.json").read_text())
comparison_clips = [("Before — original", Path(report["reference"])), (f"After — {TARGET_STYLE} (CFG {CFG:g})", Path(edited))]
sa3.compare_audio(comparison_clips, region=(START_SECONDS, END_SECONDS), title="Inpainting: original vs edited",
                  output_dir=Path(edited).parent, filename="inpaint_full_comparison")'''),
("code", r'''#@title 편집 구간만 확대 비교
sa3.compare_audio(comparison_clips, crop=(START_SECONDS, END_SECONDS), title="Inpainting: selected interval only",
                  output_dir=Path(edited).parent, filename="inpaint_region_comparison")'''),
("code", r'''#@title 검증: 구간 밖은 그대로, 구간 안은 바뀌었는지
import numpy as np, soundfile as sf
reference, sr = sf.read(report["reference"], dtype="int16", always_2d=True)
result, result_sr = sf.read(edited, dtype="int16", always_2d=True)
a, b = report["start_sample"], report["end_sample"]
assert sr == result_sr == 44100 and reference.shape == result.shape
assert np.array_equal(reference[:a], result[:a])
assert np.array_equal(reference[b:], result[b:])
assert not np.array_equal(reference[a:b], result[a:b])
print("PASS: 구간 밖 완전 보존, 구간 안 실제 변경.")'''),
("md", """## 해 보기

1. **스타일 바꾸기**: `TARGET_STYLE`을 `drum_and_bass`로 바꾸고 원본 셀부터 다시 실행합니다. 스펙트로그램에서 리듬 밀도가 어떻게 달라지나요?
2. **CFG 올리기**: `CFG = 6.0`이면 변화가 커지는 대신 음량과 왜곡도 커집니다. 플레이어의 RMS·Peak를 비교해 보세요.
3. **오디오 → 오디오 변형**: 아래 셀은 구간이 아니라 곡 전체를 다른 스타일로 바꿉니다. `NOISE_LEVEL`이 클수록 원곡에서 멀어집니다. 0.3이면 원곡 골격이 남고, 0.6이면 스타일이 바뀌고, 1.0이면 처음부터 새로 만듭니다."""),
("code", r'''#@title 오디오 → 오디오 변형 (곡 전체)
RUN_VARIATION = False  #@param {type:"boolean"}
VARIATION_PROMPT = "Instrumental warm jazz trio, electric piano, upright bass, brushed drums, no vocals"  #@param {type:"string"}
NOISE_LEVEL = 0.6  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
if RUN_VARIATION:
    varied = sa3.variation(report["reference"], VARIATION_PROMPT, noise_level=NOISE_LEVEL, seed=45)
    sa3.compare_audio([("Original", Path(report["reference"])), (f"Variation (noise {NOISE_LEVEL:g})", Path(varied))],
                      title="Audio-to-audio variation", output_dir=Path(varied).parent, filename="variation_comparison")'''),
("code", r'''#@title 저장: 결과 폴더를 zip으로 내려받기
import shutil
job_dir = str(Path(edited).parent)
zip_path = shutil.make_archive(job_dir, 'zip', job_dir)
print('저장:', zip_path, '(edited.wav가 최종 편집 파일, inpaint_full_comparison.html은 그림과 소리를 같이 봄)')
if IN_COLAB and not OUT_DIR.startswith('/content/drive'):
    from google.colab import files; files.download(zip_path)'''),
]
build("07_stable_audio_3_inpaint.ipynb", cells, "07 Stable Audio 3 Inpaint", gpu=False)
