# 슬라이드별 스펙 — 음악 분석·생성 딥러닝 입문 (목 S1~S16 · 토 S17~S30)

상세 근거와 준비 체크리스트는 `docs/lecture_plan_music_dl.md`. 이 파일은 슬라이드에 올라갈 내용만 담는다.
형식: **제목** / 본문(글머리표·표·코드) / 시각자료(플레이스홀더 캡션) / 발표자 노트.

---

## 구분 슬라이드 A — 목요일: 배경지식 (25분)
- 본문: 소리를 숫자로 → 모델이 음표·비트·악기를 읽는다 → 글에서 소리를 만든다
- 발표자 노트: 목요일은 그림으로, 토요일은 같은 흐름을 코드로.

## S1. 음악 분석·생성 딥러닝 입문
- 본문
  - 목요일: 모델이 어떻게 돌아가는지
  - 토요일: 직접 돌려보기
  - 대상: 미디어 현업 · 수식 없음
- 시각자료: [리포트 상단 흐름도 캡처: Input → Models → Output → Visual mapping]
- 발표자 노트: 이틀의 목표는 각 모델의 입력·출력과, 설정값을 바꿨을 때의 변화를 아는 것.

## S2. 모델은 소리를 어떻게 듣는가
- 본문
  - 사람: "피아노가 빨라진다"
  - 모델: 32 ms마다 88건반의 켜짐/꺼짐
  - 모델 출력 = 숫자 표 + 이벤트 목록
- 시각자료: [리포트 #s1 재생 화면 캡처 — 흰 선 현재 시각, 파란 막대 모델 출력, 노란 테두리 정답]
- 발표자 노트: 리포트 1번 장면 10초 재생. 공연 비주얼 팀에 넘기는 값을 뽑는 프로젝트에서 나온 화면이라고 소개.

## S3. 딥러닝 최소 개념 ① 데이터 → 모델 → 출력
- 본문
  - 모델 = 입력을 출력으로 바꾸는 아주 큰 함수
  - 수백만 개 숫자(파라미터)를 데이터로 맞춘 것
  - 우리는 이미 맞춰진 모델을 내려받아 쓴다 (학습 안 함)
- 시각자료: [상자 3개 다이어그램: 입력 → 모델(13.4 M / 1.4 B 파라미터 표기) → 출력]
- 발표자 노트: 파라미터 수 = 모델 크기. 피아노 채보 13.4 M, Stable Audio 3 Medium 1.4 B.

## S4. 딥러닝 최소 개념 ② 학습과 추론, 설정값
- 본문(표 2열)
  | 학습 | 추론 |
  |---|---|
  | 연구자가 GPU로 며칠 | 우리가 노트북 CPU로 초 단위 |
  | 파라미터를 바꾼다 | 파라미터는 고정, 설정값만 바꾼다 |
  | 데이터 수만 곡 | 입력 1곡 |
- 본문 하단 한 줄: 설정값(손잡이) 예 — 길이(seconds) · 프롬프트 반영 강도(CFG) · 재현(seed)
- 발표자 노트: 같은 seed와 같은 설정이면 같은 결과. 토요일에 이 설정값들을 직접 바꿔본다.

## S5. 딥러닝 최소 개념 ③ 소리를 숫자로
- 본문
  - 1초 = 44,100개 숫자 (파형)
  - 짧은 조각(프레임)마다 "어느 높이가 얼마나" → 스펙트로그램
  - 모델 출력도 프레임 단위: 31 fps, 50 fps, 86 fps
- 시각자료: [멜 스펙트로그램 1장 캡처 — 가로 시간, 세로 높이, 색 세기 (노트북 cfg_comparison.png 중 하나)]
- 발표자 노트: 리포트의 31.25 / 50 / 100 fps가 이 프레임 간격이다.

## S6. 오늘 다룰 모델 지도
- 본문(표)
  | 구분 | 모델 | 입력 → 출력 |
  |---|---|---|
  | 분석 | online AMT | 32 ms 조각 → 음표 on/off |
  | 분석 | 프레임 요약값 | 음표 목록 → 밀도·높이·onset 곡선 |
  | 분석 | Demucs | 믹스 → 6개 스템 |
  | 분석 | YourMT3+ | 믹스 → 악기별 음표(MIDI) |
  | 분석 | allin1 | 믹스 → BPM·비트·구간 |
  | 생성 | Tone Transfer / Stable Audio 3 | 흥얼거림 → 악기 / 글 → 음악 |
- 발표자 노트: 분석 5개, 생성 2개. 토요일에 직접 실행하는 것은 Stable Audio 3와 online AMT.

## S7. 실시간 피아노 채보 — online AMT
- 본문
  - 눌린 건반을 0.2초 뒤에 출력
  - onset(시작) 정확, offset(끝)은 페달에 약함
  - 비주얼은 시작 시점 + 음높이에 반응하게
- 본문(숫자 카드 4개): onset F1 0.99 / 0.90 · 지연 203 ms · CPU 4스레드 3.8배속 · 31.25 fps
- 시각자료: [리포트 #s1 스카를라티 30초 재생 화면]
- 발표자 노트: 지연 = 문맥 160 ms + 32 ms 조각 + 계산 10 ms. 비주얼은 소리와 동시가 아니라 소리에 뒤따르는 반응으로 설계. 0.99는 페달 적은 스카를라티, 0.90은 페달 많은 리스트.

## S8. 프레임 요약값 — 음표 목록의 요약
- 본문(표)
  | 값 | 쇼팽 | 바흐 | 스카를라티 | 리스트 | 비주얼 매핑 |
  |---|---|---|---|---|---|
  | 밀도 평균/최대 | 8.4 / 26 | 5.0 / 11 | 1.6 / 11 | 9.2 / 31 | 파티클 수, 색 온도 |
  | 평균 높이 중앙값 | 63.9 | 57.6 | 62.0 | 63.7 | 화면 무게중심 |
  | onset 빈도 평균/최대 | 5.2 / 19 | 3.7 / 11 | 13.0 / 21 | 9.5 / 49 | 움직임 속도 |
- 시각자료: [리포트 #s3 재생 화면]
- 발표자 노트: 모델이 아니라 모델 출력의 계산. 음표 하나하나보다 요약값이 비주얼 매핑에 쓰기 쉽다.

## S9. 스템 분리 — Demucs htdemucs_6s
- 본문
  - 완성 믹스 → vocals / drums / bass / guitar / piano / other
  - 한계: 피아노↔기타 혼동, 보컬 누출
  - 악기별 음량 곡선은 신뢰도 높음
  - 쓸모: 리믹스, 자막 싱크, 반주 제작, 드럼 히트 감지
- 시각자료: [리포트 #s4 〈Hello〉 스템 RMS envelope + 체크박스 화면]
- 인용(슬라이드 하단 작은 글씨): Demucs v4 — Rouard et al., ICASSP 2023
- 발표자 노트: 체크박스로 보컬만, 드럼만 20초 청취. 곡당 CPU 3분.

## S10. 멀티악기 채보 — YourMT3+
- 본문
  - 믹스 → 악기·음높이·시각 목록
  - 소리를 토큰 열로 바꾸는 인코더-디코더
  - 음높이·시점은 신뢰 가능, 악기 라벨은 부정확
  - 〈Hello〉 7,675음표 · 상위 라벨 Guitar(clean) 3,868 · Drums 1,298
- 시각자료: [리포트 #s5 5묶음 피아노롤 + 드럼 onset 화면]
- 발표자 노트: 4인조 록 밴드 음원에 Strings, Acoustic Piano 라벨이 나온다. 신뢰도 순서: 스템 에너지 > 드럼 onset > 비트·구조 > 묶음별 음표 > 세부 악기 라벨.

## S11. 비트·다운비트·곡 구조 — allin1
- 본문
  - BPM · 비트 · 마디 첫 박 · intro/verse/chorus 구간을 한 번에
  - 8개 모델 앙상블, CPU 곡당 약 10분
  - 6곡 BPM: 111 · 128 · 90 · 113 · 100 · 98
  - 〈다시, 다시 만나〉 비트 편차 97 ms → 수동 확인
- 시각자료: [리포트 #s6 구간 띠 + 비트 클릭 재생 화면]
- 발표자 노트: 용도 — 비트에 맞춘 컷, 장면 전환 지점, 조명·자막 큐. 다른 5곡 편차는 7~15 ms.

## S12. 생성 ① Tone Transfer — DDSP 음색 변환
- 본문
  - 입력에서 음높이 곡선 + 음량 곡선만 추출
  - 두 곡선으로 신시사이저(배음+노이즈+리버브)를 조종 (DDSP)
  - 악기 4종: Flute · Saxophone · Trumpet · Violin
  - 설정값: Octave · Loudness · Mix
- 시각자료: [Tone Transfer 웹 화면 캡처 — 라이브 데모 1분]
- 발표자 노트: 샘플 또는 즉석 녹음 5초 → Violin. Octave를 올리면 악기 음역에 맞고, Loudness를 내리면 숨소리 노이즈가 줄고, Mix는 원음 비율. Stable Audio와의 차이: 분석값으로 신시사이저를 제어하는 생성 vs 노이즈에서 전체를 만드는 생성.

## S13. 생성 ② Stable Audio 3 — 텍스트→음악
- 본문
  - 구성: 텍스트 인코더(T5Gemma) → 확산 트랜스포머 8스텝 → 오디오 디코더(SAME)
  - Medium 1.4 B 파라미터, 44.1 kHz 스테레오, 1~380초
  - 세 모드: 텍스트→오디오 / 오디오→오디오 변형 / 인페인팅·이어붙이기
  - M3 Pro: 30초 곡 약 10초
- 시각자료: [3상자 다이어그램: 텍스트 → 노이즈 8단계 정제 → 오디오 복원]
- 발표자 노트: 가중치 합 6.9 GB. 코드 MIT, 가중치는 Stability AI Community License(상업 서비스 시 확인).

## S14. 토요일에 바꿀 설정값 6개
- 본문(표)
  | 설정값 | 기본 | 올리면 | 내리면 |
  |---|---|---|---|
  | PROMPT / NEGATIVE | — | 악기·BPM·분위기 구체화 | 모델 임의 선택 |
  | CFG | 1 (편집 4) | 글에 충실, 음량·왜곡 ↑ | 1이면 negative 무시 |
  | STEPS | 8 | 느려짐, 8 이상 이득 적음 | 4 이하 뭉개짐 |
  | SEED | 42 | 다른 곡 | 같으면 같은 결과 |
  | SECONDS | 30 | 시간·메모리 ↑ | 짧게 시작 |
  | 구간 START/END · noise_level | 6~18 s · 0.6 | 변화 큼 | 원곡 많이 남음 |
- 발표자 노트: 검증 기록 — 인페인팅 CFG 4에서 RMS 0.11→0.50, 샘플 4.45 %가 풀스케일. 스펙트로그램이 밝은 것은 음량 증가일 수 있다.

## S15. 토요일 실습 내용
- 본문
  - ① 같은 글·같은 seed로 CFG 1/2/4/6 네 곡 비교
  - ② 피아노 24초 곡의 6~18초만 오케스트라로
  - ③ 완성곡을 재즈 트리오로 변형
  - ④ 분석 결과 JSON에서 비주얼용 값 추출
- 시각자료: [inpaint_full_comparison.png 캡처 — 청록 점선 편집 경계]
- 발표자 노트: ②에서 구간 밖 샘플이 보존됨을 코드로 확인한다.

## S16. 준비물
- 본문
  - 노트북(아무 OS) + 헤드폰
  - 바꿔보고 싶은 30초 이내 음원 1개 (wav/mp3)
  - Apple Silicon Mac: 배포 ZIP 설치 가능 (15 GB, 인터넷)
  - 그 외: 강사 화면 + 공개 웹 데모로 참여
- 발표자 노트: 설치는 선택. 설정값 변화를 듣는 것이 목표.

---

## 구분 슬라이드 B — 토요일: 실습 (28분)
- 본문: 노트북 2개 · 설정값 6개 · 분석 JSON 1개
- 발표자 노트: 추론 시간은 M3 Pro 실측. 백업 결과 폴더를 미리 열어둔다.

## S17. 복습
- 본문: S6 표 재사용. 오늘 실행할 항목 강조: Stable Audio 3(생성·편집), online AMT(라이브 1회), analysis.json(읽기)
- 발표자 노트: 분석은 결과 파일을 읽고, 생성은 직접 실행한다.

## S18. 환경 점검 — 노트북 첫 셀
- 본문
  - `bash scripts/start_notebook.sh` → JupyterLab
  - 커널: CTxAI · Stable Audio 3 MLX
  - 셀1 `doctor()`: 가중치 4개 SHA256 · 소스 커밋 · Metal GPU · ffmpeg
- 시각자료: [셀1 출력 JSON 캡처 — mlx 버전, device, weights_revision]
- 발표자 노트: 버전 기록이 재현의 전제. 약 5초.

## S19. 실습 A-1 — 프롬프트 쓰기 (NB01 셀3)
- 본문
  - 장르 + 악기 + 분위기 + BPM + 금지("no vocals")
  - 학습 데이터 태그 형식 권장: `TrackType: Music, VocalType: Instrumental`
  - 가사(발음 가능한 보컬)는 생성 불가
  - 고정: SECONDS 30 · SEED 42 · STEPS 8 · CFG [1,2,4,6]
- 시각자료: [셀3 코드 캡처 — 기본 프롬프트 전문]
- 발표자 노트: 기본 프롬프트(felt piano arpeggios, analog pads, 75 BPM…)를 구절별로 읽고, 수강생 제안으로 한 구절(예: 75→120 BPM)만 바꾼다.

## S20. 실습 A-2 — 같은 글, 같은 seed, CFG만 다르게 (NB01 셀4)
- 본문(코드)
  ```
  comparison_dir, cfg_clips = generate_cfg_sweep(
      PROMPT, cfg_values=CFG_VALUES, seconds=SECONDS, seed=SEED,
      steps=STEPS, negative_prompt=NEGATIVE_PROMPT)
  ```
- 본문: 30초 × 4개 ≈ 45~60초 · 실행마다 run.json / inference.log 저장
- 발표자 노트: CFG>1은 조건부·무조건부 두 번 계산이라 느리다. 대기 중 inference.log를 보여주며 별도 프로세스로 실행되고 종료 시 메모리를 반환한다고 설명. 백업: 미리 돌린 cfg_sweep 폴더.

## S21. 실습 A-3 — 멜 스펙트로그램 4장 + 플레이어 (NB01 셀6)
- 본문
  - 같은 색 범위(-90~-10 dB), 같은 축으로 비교
  - 밝기 차이는 음량 차이일 수 있음 (RMS/Peak 확인)
  - CFG 1 보수적 · 2~4 프롬프트에 근접 · 6 왜곡·음량 증가
- 시각자료: [cfg_comparison.png 캡처 2×2]
- 발표자 노트: `compare_audio(...)` 약 3초, PNG/HTML/JSON 저장. 시간이 남으면 셀8 CROP_RANGE=(5,15)로 같은 10초만 확대. 기본 권장 범위는 CFG 1~4.

## S22. 실습 B-1 — 원곡 만들기 또는 가져오기 (NB02 셀3·4)
- 본문
  - 인페인팅 = 지정 구간만 재생성, 바깥은 문맥으로 사용
  - 원곡: 24초 잔잔한 피아노 생성(seed 72) 또는 SOURCE_PATH에 내 파일
  - 설정: START 6 · END 18 · CFG 4 · SEED 73 · TARGET_STYLE orchestral_ensemble
- 시각자료: [plot_audio 파형 캡처 — 주황 구간 = 편집 범위]
- 발표자 노트: NEGATIVE_PROMPT의 "gentle solo piano…"는 원곡 성격을 억제하기 위한 것. CFG 1에서는 무시되므로 4를 쓴다.

## S23. 실습 B-2 — 6~18초만 오케스트라로 (NB02 셀6)
- 본문(코드)
  ```
  edited = inpaint(source, EDIT_PROMPT, start=6.0, end=18.0,
                   seed=73, steps=8, cfg=4.0,
                   negative_prompt=NEGATIVE_PROMPT, crossfade=0.10)
  ```
- 본문: 한 호출 = ffmpeg 변환 → 모델 인페인팅 → 구간만 합성(0.1초 크로스페이드) → edit_report.json
- 발표자 노트: 24초 입력 약 8~10초. 공식 가이드: 구간이 짧으면 문맥이 강해 원곡과 거의 같게 나오므로 처음엔 넓게 잡는다. 변주: TARGET_STYLE="drum_and_bass" 또는 CFG=6.

## S24. 실습 B-3 — 전후 비교와 보존 검증 (NB02 셀8·10·12)
- 본문
  - 전체 비교: 청록 점선 = 편집 경계
  - 구간만 확대 비교: 음색·리듬 밀도·저음/고음 분포
  - assert: 구간 밖 샘플 완전 동일, 구간 안 실제 변경
  - 산출물: inpaint_full_comparison.html (오디오 내장, 공유 가능)
- 시각자료: [inpaint_region_comparison.png 캡처]
- 발표자 노트: 편집 결과에 검증 코드를 함께 둔다. 셀12 출력 `PASS: 구간 밖 완전 보존, 구간 안 실제 변경.`

## S25. 실습 C — 오디오→오디오 변형 + 수강생 음원
- 본문
  - noise_level: 올릴수록 멜로디·리듬부터 지워지고 음색·조성만 남음
  - 0.4 악기만 교체 · 0.6 스타일 교체 · 1.0 완전 재생성
  - `variation(cfg_clips[0][1], "warm jazz trio…", noise_level=0.6, seed=45)`
  - 내 음원: inputs/에 넣고 SOURCE_PATH · START/END 수정
- 발표자 노트: 30초 약 10초. 가능하면 0.3으로 한 번 더 돌려 A/B. 380초 초과 파일은 거부되므로 미리 자른다.

## S26. 실습 D-1 — 실시간 피아노 채보 라이브 (서버 CPU)
- 본문(코드)
  ```
  CUDA_VISIBLE_DEVICES="" python run_piano_amt.py scarlatti.flac \
      --max_seconds 30 --threads 4 --ref_midi scarlatti.midi
  ```
- 본문: 30초 클립 약 8초 · amt_timing.json의 onset_f1 · amt_roll.png
- 시각자료: [amt_roll.png 캡처]
- 발표자 노트: `--no-delay-compensation`을 붙여 다시 돌리면 onset F1이 0.99에서 0.2대로 떨어진다. S7의 지연 보정 설명과 연결.

## S27. 실습 D-2 — 밴드 분석 JSON 열기
- 본문(코드)
  ```
  d = json.load(open('band/outputs/02_hello/analysis.json'))
  d['tempo']['bpm'], len(d['tempo']['beats'])        # 128, 476
  [(round(s['start']), s['label']) for s in d['structure']]
  sum(x > 0.5 for x in d['stems']['drums']['onset_strength'])
  ```
- 본문: 구조 intro → verse → inst → verse → chorus → solo → chorus → outro
- 시각자료: [structure_map.png 캡처]
- 발표자 노트: Demucs·YourMT3+·allin1은 곡당 3~10분이라 수업 중에는 실행하지 않는다. 리포트 #s4·#s6을 옆에 띄워 숫자와 그래프를 대응시킨다. 스템 청취는 docs/audio/hello_stems.

## S28. 분석 모델 설정값 요약
- 본문(표)
  | 모델 | 바꾸는 값 | 효과 |
  |---|---|---|
  | Demucs | `-n htdemucs_6s` / `--two-stems vocals` / `--shifts` | 스템 수 · 속도 · 아티팩트 |
  | allin1 | `--model harmonix-fold0` / `--activ` | 8배 빠름, 다운비트 정확도 소폭 하락 |
  | YourMT3+ | 체크포인트 `-m` / `OMP_NUM_THREADS` | 라벨 품질 · 속도 |
  | online AMT | `--threads` / `--mode` / 지연 보정 | 실시간 여유 · F1 |
- 발표자 노트: 개인 실행용 참고. 전부 CPU 기준.

## S29. 정리 — 생성 모델 설정값 요약
- 본문: S14 표의 "올리면/내리면" 열을 빈 칸으로 두고 수업 중 청취 결과로 채움
- 발표자 노트: 수강생이 말한 표현을 그대로 적는다.

## S30. 이후 과제 + 링크
- 본문
  - Tone Transfer: 녹음 → 악기 변환
  - HF Space stabilityai/stable-audio-3 로 프롬프트 연습
  - Apple Silicon: 배포 ZIP `bash scripts/setup.sh`
  - 리포트 HTML에서 섹션별 소리와 값 다시 보기
- 본문(링크): github.com/Stability-AI/stable-audio-3 · magenta.withgoogle.com/tone-transfer · github.com/mir-aidj/all-in-one · github.com/facebookresearch/demucs
- 발표자 노트: 질문 받기.
