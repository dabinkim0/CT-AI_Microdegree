# 미디어 현업자를 위한 음악 딥러닝 입문 — 수업 계획서 (목요일 배경 / 토요일 실습)

작성일: 2026-10-19 · 담당 분량: 하루 20~30분 × 2일 · 한 슬라이드 덱에 연속 배치 (목요일 S1~S16, 토요일 S17~S30)

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 대상 | 영상·방송·광고·공연 등 미디어 현업 숙련자. ML 비전공. 수식 없음. 각 모델의 입력·출력과 설정값 변경 효과 중심 |
| 목요일 (25분) | 딥러닝 최소 개념 + 토요일에 다룰 7개 모델의 입력→출력·용도·설정값(하이퍼파라미터) 소개 |
| 토요일 (28분) | Stable Audio 3 노트북 2개를 실제로 실행(생성·CFG 비교·인페인팅·변형) + 분석 모델 실행 결과를 코드로 열어보기 |
| 다루는 모델 | 분석: online AMT(실시간 피아노 채보), 프레임 요약값, Demucs(스템 분리), YourMT3+(멀티악기 채보), allin1(비트·구조) · 생성: Tone Transfer(DDSP 음색 변환, 웹 데모), Stable Audio 3 Medium(텍스트→음악, 인페인팅, 오디오→오디오) |
| 제외 | 리포트 2번 멜로디 추출(CHIME) — 수업에서 다루지 않음 |
| 핵심 자료 | ① `docs/report_2026-10-06.html` (또는 standalone 버전) 브라우저로 열어 섹션 `#s1,#s3~#s7` 사용 ② https://magenta.withgoogle.com/tone-transfer (실제 앱은 sites.research.google/tonetransfer 로 연결) ③ `model.zip` → `model/notebooks/01_generate_music.ipynb`, `02_inpaint_music.ipynb` |
| 실행 하드웨어 | 토요일 Stable Audio 3는 **Apple Silicon Mac에서만** 실행 (패키지가 MLX 기반. 검증 기기 M3 Pro 18 GB). 분석 모델은 연구실 Linux 서버 **CPU**로 사전 실행 + 피아노 채보 1개만 라이브. GPU는 사용하지 않음 |

덱 전체 흐름: 소리를 숫자로 바꾼다 → 모델이 음표·비트·악기를 읽는다 → 글에서 소리를 만든다. 목요일은 그림으로, 토요일은 코드로.

---

## 1. 목요일 — 타임라인 (합계 25분, 버퍼 포함 30분)

| 시각 | 길이 | 블록 | 슬라이드 |
|---|---|---|---|
| 0:00 | 2분 | 도입: 모델은 소리를 어떻게 듣는가 | S1~S2 |
| 2:00 | 4분 | 딥러닝 최소 개념 3장 (데이터→모델→출력 / 학습 vs 추론 / 소리를 숫자로) | S3~S5 |
| 6:00 | 2분 | 오늘 다룰 모델 지도 (분석 5 + 생성 2) | S6 |
| 8:00 | 4분 | 실시간 피아노 채보 + 프레임 요약값 | S7~S8 |
| 12:00 | 3분 | 스템 분리 (Demucs) | S9 |
| 15:00 | 2분 | 멀티악기 채보 (YourMT3+) | S10 |
| 17:00 | 2분 | 비트·다운비트·곡 구조 (allin1) | S11 |
| 19:00 | 2분 | 생성 ①: Tone Transfer 웹 데모 (DDSP) | S12 |
| 21:00 | 3분 | 생성 ②: Stable Audio 3 (텍스트→음악, 인페인팅) + 설정값 미리보기 | S13~S14 |
| 24:00 | 1분 | 토요일 안내 + 준비물 | S15~S16 |
| 25:00 | 5분 | 질문/버퍼 | — |

시간이 모자라면 자르는 순서: S10(YourMT3+)을 S9 끝에 한 줄로 흡수 → S8(프레임 요약값)을 S7에 합침 → S5(소리를 숫자로)를 S4와 합침.

## 2. 목요일 — 슬라이드별 개요

각 항목: **제목 / 핵심 메시지 / 쓸 자료(데모·그림·코드) / 말할 내용 메모**

### S1. 표지 — 음악 분석·생성 딥러닝 입문
- 핵심: 이틀 동안 할 일 한 문장. 목요일 = 어떻게 돌아가는지, 토요일 = 직접 돌려보기.
- 자료: 리포트 상단 흐름도(① Input → ② Models → ③ Output → ④ Visual mapping) 캡처.

### S2. 도입 — 모델은 소리를 어떻게 듣는가
- 핵심: 사람은 "피아노가 빨라진다"고 듣고, 모델은 "32 ms마다 88개 건반의 켜짐/꺼짐"으로 듣는다. 모델의 출력은 결국 **숫자 표와 이벤트 목록**이다.
- 자료: 리포트 `#s1` 장면을 10초 재생 (흰 선 = 현재, 파란 막대 = 모델 출력, 노란 테두리 = 정답). 소리와 막대가 같이 움직이는 것을 보여주고 끝.
- 메모: 공연 비주얼 팀에 넘길 값을 뽑는 프로젝트에서 나온 화면이라고 소개.

### S3. 딥러닝 최소 개념 ① — 데이터 → 모델 → 출력
- 핵심: 모델 = 입력을 출력으로 바꾸는 아주 큰 함수. 안에 있는 수백만 개 숫자(파라미터)를 데이터로 맞춘 것. 우리는 **이미 맞춰진 모델을 내려받아 쓰기만 한다**(학습 안 함).
- 자료: 상자 3개 그림(입력 상자 → 모델 상자(파라미터 수 표기: AMT 13.4 M, CHIME 1.4 M, Stable Audio 3 Medium 1.4 B) → 출력 상자).
- 메모: 파라미터 수 = 모델 크기. 100배 차이만 언급.

### S4. 딥러닝 최소 개념 ② — 학습과 추론, 설정값
- 핵심: 학습은 연구자가 GPU로 며칠, 추론은 우리가 노트북 CPU로 초 단위. 우리가 돌리는 건 전부 추론. 추론에도 결과를 바꾸는 설정값(하이퍼파라미터)이 있고, 토요일에 이를 직접 바꿔본다.
- 자료: 2열 표 (학습 | 추론) + 설정값 예시 3개: 길이(seconds), 프롬프트 반영 강도(CFG), 재현(seed).
- 메모: 같은 seed와 같은 설정이면 같은 결과.

### S5. 딥러닝 최소 개념 ③ — 소리를 숫자로: 파형 → 스펙트로그램 → 프레임
- 핵심: 1초 = 44,100개 숫자(파형). 모델은 보통 이를 짧은 조각(프레임)마다 "어느 높이의 소리가 얼마나 있나"로 바꾼 그림(스펙트로그램)을 본다. 그래서 모델 출력도 **프레임 단위**(31 fps, 50 fps, 86 fps…)로 나온다.
- 자료: 토요일 노트북이 그리는 멜 스펙트로그램 1장 (`model/reports/01_generate_music.executed.ipynb`의 `cfg_comparison.png` 중 하나 캡처). 가로 = 시간, 세로 = 높이, 색 = 세기.
- 메모: 리포트의 31.25 / 50 / 100 fps가 이 프레임 간격이다.

### S6. 오늘 다룰 모델 지도
- 핵심: 분석(소리 → 뜻) 5개와 생성(뜻 → 소리) 2개. 각 줄은 "입력 → 출력" 한 줄.
- 자료: 아래 표를 슬라이드로.

| 구분 | 모델 | 입력 → 출력 | 토요일 |
|---|---|---|---|
| 분석 | online AMT (피아노 채보) | 마이크/파형 32 ms 조각 → 음표 on/off (pitch, 시각) | 라이브 1회 |
| 분석 | 프레임 요약값 | 음표 목록 → 밀도·평균 높이·onset 빈도 곡선 | 결과 열어보기 |
| 분석 | Demucs | 믹스 wav → 6개 스템 wav (vocals/drums/bass/guitar/piano/other) | 사전 실행 결과 청취 |
| 분석 | YourMT3+ | 믹스 wav → 악기 라벨 붙은 음표 목록(MIDI) | 결과 열어보기 |
| 분석 | allin1 | 믹스 wav → BPM, 비트, 다운비트, 구간 라벨 | 결과 열어보기 |
| 생성 | Tone Transfer (DDSP) | 흥얼거림 → 플루트/색소폰/트럼펫/바이올린 소리 | 웹 데모 (목요일 시연) |
| 생성 | Stable Audio 3 Medium | 글(+기존 오디오, 구간) → 44.1 kHz 스테레오 음악 | 노트북 2개 실행 (핵심) |

### S7. 실시간 피아노 채보 — online AMT (리포트 1번)
- 핵심: 눌린 건반을 0.2초 뒤에 출력하는 모델. 정확도는 onset(시작)은 높고 offset(끝)은 페달 때문에 낮다 → 비주얼은 **시작 시점과 음높이**에 반응하게 설계.
- 자료: 리포트 `#s1` 재생(스카를라티 30초). 숫자 카드 4개만: onset F1 0.99(스카를라티)/0.90(리스트), 지연 203 ms, CPU 4스레드 3.8배속, 31.25 fps.
- 메모: 지연 = 문맥 확보 160 ms + 32 ms 조각 + 계산 10 ms. 비주얼은 소리와 동시가 아니라 뒤따르는 반응으로 설계.

### S8. 프레임 요약값 — 음표 목록의 요약 (리포트 3번)
- 핵심: 음표 하나하나보다 **동시에 울리는 수(밀도), 평균 높이, 최근 1초 onset 수** 같은 요약 곡선이 비주얼·연출에는 더 쓰기 쉽다. 모델이 아니라 모델 출력의 간단한 계산.
- 자료: 리포트 `#s3` 재생 + 표(쇼팽/바흐/스카를라티/리스트 밀도 평균·최대, 비주얼 매핑 예: 파티클 수, 화면 무게중심, 움직임 속도).
- 메모: 모델 출력을 그대로 쓰기보다 요약값으로 바꿔 쓰는 것이 실무에서 흔한 패턴.

### S9. 스템 분리 — Demucs htdemucs_6s (리포트 4번)
- 핵심: 완성된 믹스를 보컬/드럼/베이스/기타/피아노/기타(other) 6트랙으로 되돌린다. 한계는 피아노↔기타 혼동과 보컬 누출. 악기별 음량 곡선은 신뢰도가 높다.
- 자료: 리포트 `#s4`에서 〈Hello〉 스템 체크박스를 켜고 끄며 20초 청취(보컬만, 드럼만). RMS envelope 그래프.
- 메모: 용도 — 리믹스, 가사 자막 타이밍, 보컬 제거 반주, 드럼 히트 감지. 곡당 CPU 3분.

### S10. 멀티악기 채보 — YourMT3+ (리포트 5번)
- 핵심: 믹스에서 악기·음높이·시각 목록을 내는 모델. 소리를 토큰 열로 바꾸는 인코더-디코더 구조. 음높이·시점은 신뢰 가능하지만 악기 라벨은 부정확(4인조 록 밴드 음원에 Strings, Acoustic Piano가 나옴).
- 자료: 리포트 `#s5` 재생 + 곡 표(음표 수, 상위 악기 라벨, CPU 처리 시간). 실측: 〈Hello〉 7,675음표, 상위 라벨 Guitar(clean) 3,868 · Drums 1,298 · Guitar(distortion) 847.
- 메모: 신뢰도 순서 — 스템 에너지 > 드럼 onset > 비트·구조 > 묶음별 음표 > 세부 악기 라벨.

### S11. 비트·다운비트·곡 구조 — allin1 (리포트 6번)
- 핵심: BPM, 비트 시각, 마디 첫 박, intro/verse/chorus 구간을 한 번에. 8개 모델 앙상블이라 CPU로 곡당 약 10분. 결과는 이벤트 목록 + 100 fps 확률 곡선.
- 자료: 리포트 `#s6` 재생(클릭 소리 + 구간 띠). 표: 6곡 BPM(111/128/90/113/100/98), 〈다시, 다시 만나〉 비트 간격 편차 97 ms → 수동 확인 필요.
- 메모: 용도 — 비트에 맞춘 컷 편집, 장면 전환 지점, 가사·조명 큐.

### S12. 생성 ① — Tone Transfer: DDSP 음색 변환
- 핵심: 입력에서 **음높이 곡선과 음량 곡선만** 뽑고(SPICE 모델), 그 두 곡선으로 신시사이저(배음 + 노이즈 + 리버브)를 조종해 다른 악기 소리를 낸다. 즉 분석 후 재합성. 모델이 작아 브라우저에서 실행된다.
- 자료: 웹 데모 1분 시연 — 샘플(또는 즉석 녹음 5초) → Violin 선택 → 재생. 하단 왼쪽 ML 컨트롤 패널의 **Octave / Loudness / Mix** 슬라이더를 한 번씩 움직여 차이를 듣는다.
- 메모: Octave를 올리면 악기 음역에 맞아 자연스러워지고(목소리는 보통 바이올린보다 낮음), Loudness를 내리면 숨소리·노이즈가 줄고, Mix는 원음과 섞는 비율. Stable Audio와의 차이: 분석값으로 신시사이저를 제어하는 생성 vs 노이즈에서 전체를 만드는 생성.

### S13. 생성 ② — Stable Audio 3: 텍스트→음악
- 핵심: 글(프롬프트)을 이해하는 언어 모델(T5Gemma) + 노이즈를 8단계에 걸쳐 음악의 압축 표현(latent)으로 다듬는 확산 모델(DiT) + 압축을 풀어 44.1 kHz 스테레오로 되돌리는 디코더(SAME). 세 구성요소의 가중치 합 6.9 GB.
- 자료: 3상자 그림(텍스트 인코더 → 확산 트랜스포머 ×8 스텝 → 오디오 디코더). 세 가지 모드 표: 텍스트→오디오 / 오디오→오디오(변형) / 인페인팅(구간만 다시 만들기, 이어 붙이기 continuation).
- 메모: Medium = 1.4 B 파라미터, 최대 380초. M3 Pro에서 30초 곡 약 10초(CFG 1). 라이선스: 코드 MIT, 가중치는 Stability AI Community License(상업 서비스 시 확인).

### S14. 토요일에 바꿀 설정값 6개 (Stable Audio 3)
- 핵심: 각 설정값이 결과를 어떻게 바꾸는지 미리 정리.
- 자료: 아래 표 그대로.

| 설정값 | 범위(기본) | 올리면 | 내리면 |
|---|---|---|---|
| `PROMPT` / `NEGATIVE_PROMPT` | 자유 문장 / 빈칸 | 악기·BPM·분위기·"no vocals"를 구체적으로 쓸수록 의도에 가까움 | 짧으면 모델이 임의로 채움 |
| `CFG` (글을 따르는 세기) | 0~10 (생성 1, 인페인팅 4. 공식 문서는 강한 준수에 7 제안) | 글에 충실해지지만 **음량이 커지고 왜곡·클리핑** 증가(검증 기록: 인페인팅 CFG 4에서 RMS 0.11→0.50, 4.45 % 샘플 풀스케일) | 1이면 negative prompt 무시, 가장 부드럽고 빠름 |
| `STEPS` | 1~32 (8) | 느려짐. 공식 문서: 8이 sweet spot, 그 이상은 품질이 꼭 좋아지지 않음 | 1이면 한 번의 forward(가장 빠름), 4 이하는 뭉개짐 |
| `SEED` | 정수 (42) | 다른 곡 | 같은 seed + 같은 설정 = 같은 결과 |
| `SECONDS` | 1~380 (30) | 길수록 시간·메모리 ↑ (30→60→120 순으로) | 짧게 시작 |
| 인페인팅 `START/END`, `CROSSFADE` | 0.2초 이상 / 0.1초 | 구간이 넓을수록 원곡 문맥 영향 ↓, 변화 큼 | 좁으면 원곡이 강하게 남음 |
| 변형 `noise_level` | 0.01~1 (0.6) | 1에 가까울수록 원곡에서 멀어짐 | 0.3이면 원곡 골격 유지 |

### S15. 토요일 실습 내용
- 핵심: ① 같은 글·같은 seed로 CFG 1/2/4/6 네 곡 만들어 듣고 비교 ② 피아노 24초 곡의 6~18초만 오케스트라로 바꾸기(나머지 샘플은 1비트도 안 바뀜을 코드로 확인) ③ 완성곡을 재즈 트리오로 변형 ④ 분석 모델 결과 JSON을 파이썬으로 열어 비주얼용 값 추출.
- 자료: 토요일 산출물 예시 — `inpaint_full_comparison.png`(편집 경계 청록 점선) 1장.

### S16. 준비물 (수강생)
- 핵심: 노트북(아무 OS), 헤드폰, 본인이 바꿔보고 싶은 **30초 이내 음원 1개**(wav/mp3). Apple Silicon Mac이면 강사 배포 ZIP 설치 가능(15 GB 여유, 인터넷). 그 외는 강사 화면 + 공개 웹 데모(HF Space `stabilityai/stable-audio-3`)로 참여.
- 메모: 설치는 선택. 토요일 목표는 강사 Mac 화면을 보며 설정값 변화를 듣는 것.

---

## 3. 모델·태스크별 카드 (슬라이드 작성과 토요일 멘트용 정리)

### 3-1. online AMT — 실시간 피아노 채보 (리포트 1번, `piano/tools/run_piano_amt.py`)
- 한 줄: 피아노 소리를 32 ms마다 듣고 88건반 각각의 상태(off/offset/sustain/onset/re-onset)를 내는 인과(과거만 보는) 모델. PARpiano의 전신 online_amt(ISMIR 2020), 13.4 M 파라미터.
- 입력 → 출력: 16 kHz 모노 파형 512샘플 조각 → 프레임마다 88×5 상태 → 음표 이벤트 `{onset, offset, pitch, velocity(RMS 추정)}` + 피아노롤.
- 현업 쓸모: 라이브 공연 비주얼 트리거, 연주 연습 피드백, 악보 초안. CPU 노트북으로 실시간 가능(4스레드 4배속).
- 토요일 설정값과 효과:
  - `--threads 1→4`: 속도 2.0배→4.1배. 정확도 불변. 실시간 여유가 달라짐.
  - `--mode online|offline`: offline은 앞뒤 문맥을 다 보는 대신 실시간 불가. F1 차이 0.2점 이내.
  - `--no-delay-compensation`: 160 ms 보정을 끄면 onset F1 0.99→0.27. 지연 보정의 필요성을 보여주는 데모.
  - `--silence_threshold 0.05 / --silence_patience 100`: 무음 게이트. 낮추면 여린 음을 잡지만 소음도 잡음.
  - 곡 선택: 스카를라티(페달 적음) vs 리스트(페달 많음)의 offset F1 0.93 vs 0.02.

### 3-2. 프레임 요약값 (리포트 3번, `piano/tools/frame_summary.py`)
- 한 줄: 음표 목록을 31.25 fps 곡선 5개(`density, mean_pitch, pitch_spread, onset_rate_1s, active_pitches`)로 요약.
- 입력 → 출력: `amt_notes.json` → `frame_summary.json`.
- 현업 쓸모: 비주얼 파라미터(파티클 수, 색 온도, 화면 무게중심, 움직임 속도)에 바로 매핑. 곡별 범위를 미리 알 수 있어 리허설 전에 스케일을 잡는다.
- 토요일 설정값: onset 빈도의 창 길이(1초 → 0.5/2초): 짧으면 민감·떨림, 길면 부드럽고 느림. 비주얼 쪽 이동평균(100 ms) 적용 전후 비교.

### 3-3. Demucs htdemucs_6s — 스템 분리 (리포트 4번, `band/tools/features.py` 앞단)
- 한 줄: 믹스 파형을 받아 악기별 파형을 내는 U-Net+Transformer 하이브리드.
- 인용: Rouard, Massa & Défossez, "Hybrid Transformers for Music Source Separation," ICASSP 2023 (Demucs v4 htdemucs). 전신: Défossez, "Hybrid Spectrogram and Waveform Source Separation," ISMIR 2021 MDX Workshop (Demucs v3). 코드 github.com/facebookresearch/demucs (MIT).
- 입력 → 출력: 44.1 kHz 스테레오 wav → `vocals, drums, bass, guitar, piano, other` wav 6개 → 50 fps `rms_norm`, `onset_strength(drums)`, `spectral_centroid_hz`.
- 현업 쓸모: 가사 자막 싱크, 반주 제작, 샘플링, 악기별 레이어 연출.
- 토요일 설정값과 효과(명령줄):
  - `-n htdemucs`(4스템) vs `htdemucs_6s`(6스템, 기타·피아노 추가, 혼동 가능) vs `htdemucs_ft`(미세조정, 약 4배 느리고 조금 더 깨끗).
  - `--two-stems vocals`: 보컬/반주 2개만. 가장 빠름.
  - `--shifts 1→5`: 랜덤 시프트 평균으로 아티팩트 감소, 시간은 비례 증가.
  - `--overlap 0.25`, `--segment`: 메모리 부족 시 줄임.
  - `-d cpu`, `--mp3`: 서버 CPU 기준 4분 곡 약 3분.

### 3-4. YourMT3+ — 멀티악기 채보 (리포트 5번, `band/tools/run_yourmt3.py`)
- 한 줄: 믹스 스펙트로그램을 2초 조각으로 잘라 "악기·음·시각" 토큰 열로 번역하는 인코더-디코더(Perceiver-TF + MoE + T5 디코더). 체크포인트 YPTF.MoE+Multi (noPS).
- 입력 → 출력: 16 kHz 믹스 → `notes.json` `{onset, offset, pitch, program, is_drum, instrument}` + MIDI. 보조: 악기별 `note_onset_density`(50 fps).
- 현업 쓸모: 커버·리믹스용 MIDI 초안, 어느 악기가 어느 음역에서 얼마나 바쁜지(활동량) 시각화.
- 토요일 설정값: `-m` 체크포인트 교체(YMT3+, YPTF+Single 등), `--device cpu` + `OMP_NUM_THREADS=8`(8스레드 RTF 1.0 → 4분 곡 4분), 세부 program 라벨 대신 `notes_coarse`(drums/bass/guitar/vocals/keys_other 5묶음) 사용. 효과: 묶음별로 보면 안정적, 세부 라벨에는 Strings 등 오류가 섞임.

### 3-5. allin1 — 비트·다운비트·곡 구조 (리포트 6번)
- 한 줄: 내부에서 Demucs로 스템을 나눈 뒤 이웃 어텐션 모델 8개 앙상블로 비트·다운비트·구간을 동시에 예측.
- 입력 → 출력: wav → `{bpm, beats[], downbeats[], beat_positions[], segments[{start,end,label}]}` + `--activ`로 100 fps `beat/downbeat/segment` 확률 곡선.
- 현업 쓸모: 비트 싱크 컷, 장면 전환 추천, 조명·자막 큐, 구조 지도 그림(`structure_map.png`).
- 토요일 설정값: `-d cpu`(필수), `--model harmonix-all`(8모델, 곡당 10분) vs `harmonix-fold0`(1모델, 8배 빠름, 약간 덜 안정), `--activ`(곡선 저장), `--keep-byproducts`(스템 보관), `--visualize/--sonify`(그림·클릭 wav). 효과: 단일 모델로 바꿔도 BPM은 거의 같고 다운비트 정확도가 소폭 하락. 템포 흔들리는 곡(비트 간격 편차 97 ms)은 수동 확인.

### 3-6. Tone Transfer — DDSP 음색 변환 (웹 데모)
- 한 줄: 입력에서 음높이(f0)·음량 곡선을 뽑아 미분 가능한 신시사이저를 조종, 네 악기(Flute/Saxophone/Trumpet/Violin) 소리로 재합성.
- 입력 → 출력: 15초 내외 단선율 녹음/업로드/샘플 → 같은 길이 악기 소리.
- 현업 쓸모: 가이드 멜로디를 악기로 바꿔 데모 제작, 효과음 스케치.
- 설정값과 효과: 악기 선택(음역이 맞는 악기가 자연스러움), **Octave**(목소리→바이올린은 +1이 보통 자연스러움), **Loudness**(낮추면 숨소리성 노이즈 감소), **Mix**(원음 비율). 다성(화음) 입력은 처리 불가. 단선율만.

### 3-7. Stable Audio 3 Medium (MLX) — 생성·편집 (토요일 핵심, `model/notebooks/*`)
- 한 줄: 텍스트 조건 잠재 확산 모델. 8스텝 증류, 44.1 kHz 스테레오, 1~380초.
- 입력 → 출력: ① 프롬프트 → wav ② 프롬프트 + 기존 wav + `noise_level` → 변형 wav ③ 프롬프트 + 기존 wav + 구간 → 구간만 새로 만든 wav(구간 밖은 PCM 샘플 단위 보존, 안쪽 0.1초 크로스페이드).
- 현업 쓸모: BGM 시안 다량 생성, 기존 BGM의 특정 구간만 분위기 교체(컷 길이 맞추기), 곡 연장(continuation), 레퍼런스 곡의 스타일 변형.
- 설정값(위 S14 표) + 패키지 제약: Mac 전용(MLX), 추론마다 별도 프로세스(Metal 메모리 반환), 실행마다 `outputs/<시각>_<모드>_<id>/`에 `run.json`(프롬프트·시드·명령·시간), `inference.log`, wav, 비교 PNG/HTML/JSON 저장. 실행 기록이 자동으로 남는다.

---

## 4. 토요일 — 타임라인 (합계 28분, 버퍼 포함 30분)

| 시각 | 길이 | 블록 | 슬라이드 | 실행 |
|---|---|---|---|---|
| 0:00 | 2분 | 복습 1장 + 환경 점검 | S17~S18 | NB01 셀1 `doctor()` |
| 2:00 | 7분 | 실습 A: 텍스트→음악, CFG 1/2/4/6 비교 | S19~S21 | NB01 셀3·4·6 (+8) |
| 9:00 | 8분 | 실습 B: 인페인팅 — 피아노 6~18초를 오케스트라로 | S22~S24 | NB02 셀3·4·6·8·10·12 |
| 17:00 | 3분 | 실습 C: 오디오→오디오 변형 + 수강생 음원 1개 | S25 | NB01 셀10 / NB02 셀3 `SOURCE_PATH` |
| 20:00 | 6분 | 실습 D: 분석 모델 결과를 코드로 열기 (피아노 채보 라이브 1회 + JSON 읽기) | S26~S28 | 서버 터미널 + 짧은 파이썬 |
| 26:00 | 2분 | 정리: 설정값 요약, 이후 과제 | S29~S30 | — |
| 28:00 | 2분 | 버퍼 | — | — |

시간이 모자라면 자르는 순서: 실습 C → 실습 D의 라이브 채보(사전 실행 PNG로 대체) → 실습 A의 CFG 4개를 2개(1, 4)로.

## 5. 토요일 — 슬라이드별 개요 + 실행 셀 목록

추론 시간은 `model/reports/VALIDATION.md`의 M3 Pro 18 GB 실측(2026-10-16)이다. 다른 Mac에서는 리허설로 다시 잰다.

### S17. 복습
- 핵심: S6 표를 다시 띄우고 오늘 실행할 항목에 표시. 분석은 결과 파일을 읽고, 생성은 직접 실행한다.

### S18. 환경 점검 — 노트북 첫 셀
- 핵심: 노트북은 환경 확인 셀부터. 가중치 4개 SHA256·소스 커밋·Metal GPU·ffmpeg를 한 번에 검사.
- 실행: 터미널 `bash scripts/start_notebook.sh` → JupyterLab → `01_generate_music.ipynb` 열기 → 커널 **CTxAI · Stable Audio 3 MLX** → 셀1 실행 (≈5초). 출력 JSON(mlx 버전, device, weights_revision)을 가리키며 버전 기록이 재현의 전제라고 설명.
- 자료: 셀1 출력 캡처(백업용).

### S19. 실습 A-1 — 프롬프트 쓰기 (NB01 셀3)
- 핵심: 좋은 프롬프트 = 악기 + 템포(BPM) + 분위기 + 믹스 지시 + 금지("no vocals"). 기본 프롬프트(felt piano arpeggios, analog pads, 75 BPM, cinematic ambient…)를 한 구절씩 읽으며 설명.
- 공식 프롬프트 가이드 팁(`stable-audio-3/docs/guides/prompting.md`): 장르·악기·분위기·BPM 네 요소를 넣는다. 학습 데이터(Freesound, AudioSparx) 태그 형식 `TrackType: Music, VocalType: Instrumental`을 앞에 붙이면 품질·일관성이 올라가고, 솔로 악기는 `TrackType: Instrument`로 시작. 모델은 알아들을 수 있는 가사를 내지 못한다(보컬 비슷한 질감만).
- 실행: 셀3 값 확인. 수강생 제안 1개로 `PROMPT` 한 구절만 교체(예: 75 BPM → 120 BPM). `SECONDS=30, SEED=42, STEPS=8, CFG_VALUES=[1,2,4,6]` 유지.
- 확인 포인트: BPM 숫자를 모델이 실제로 따르는지 다음 슬라이드에서 확인.

### S20. 실습 A-2 — 같은 글, 같은 seed, CFG만 다르게 (NB01 셀4)
- 핵심: CFG는 프롬프트 반영 강도. 네 번 순차 생성, 각 실행 폴더에 `run.json`이 남는다.
- 실행: 셀4 `generate_cfg_sweep(...)` (30초×4개 ≈ 45~60초. CFG>1은 조건부·무조건부 두 번 계산이라 더 느림). 기다리는 동안 `inference.log`가 셀에 흘러나오는 것을 보여주며 별도 프로세스로 실행되고 종료 시 메모리를 반환한다고 설명.
- 백업: 미리 돌린 `outputs/<시각>_cfg_sweep_*/` 폴더 또는 `reports/01_generate_music.executed.ipynb`(오디오 포함 29 MB)를 열어둔다.

### S21. 실습 A-3 — 멜 스펙트로그램 4장 + 플레이어 (NB01 셀6, 선택 셀8)
- 핵심: 같은 색 범위(-90~-10 dB), 같은 축으로 비교. 밝기 차이는 음량 차이일 수 있음(플레이어 밑 RMS/Peak 숫자). CFG 6에서 왜곡·클리핑이 들리는지 확인.
- 실행: 셀6 `compare_audio(...)` (≈3초) → PNG/HTML/JSON 저장. 시간 남으면 셀8 `RUN_CROP_COMPARISON=True, CROP_RANGE=(5,15)`로 같은 10초만 확대.
- 정리(판서): CFG 1 = 보수적 / 2~4 = 프롬프트에 근접 / 6 = 왜곡·음량 증가. 기본 권장 범위 1~4.

### S22. 실습 B-1 — 원곡 만들기 또는 가져오기 (NB02 셀3·4)
- 핵심: 인페인팅은 지정 구간만 재생성하고 바깥은 문맥으로 쓴다. 먼저 24초 잔잔한 피아노를 생성(`SOURCE_PROMPT`, seed 72) 하거나 `SOURCE_PATH`에 본인 파일.
- 실행: `02_inpaint_music.ipynb` 열기 → 셀1(≈5초) → 셀3 설정 읽기(`START=6, END=18, TARGET_STYLE="orchestral_ensemble", CFG=4, SEED=73, NEGATIVE_PROMPT=...`) → 셀4 (24초 생성 ≈ 6~8초) → 파형 그림의 주황 구간 = 편집 범위.
- 메모: `NEGATIVE_PROMPT`의 "gentle solo piano…"는 원곡 성격을 억제하기 위한 것. CFG 1에서는 무시되므로 4를 쓴다.

### S23. 실습 B-2 — 6~18초만 오케스트라로 (NB02 셀6)
- 핵심: 하나의 함수 호출로 ① ffmpeg 변환(44.1 kHz 스테레오 PCM16) ② 모델 인페인팅 ③ 구간만 잘라 원곡에 합성(0.1초 크로스페이드) ④ `edit_report.json`.
- 실행: 셀6 `inpaint(...)` (24초 입력 ≈ 8~10초). 출력 JSON의 `outside_region_bit_exact: true`, `edited_region_changed: true` 가리키기.
- 공식 가이드: 구간이 짧으면 문맥이 강해 원곡과 거의 같게 나오므로 처음엔 넓게 마스킹하고 줄여 간다. 프롬프트는 주변 문맥과 이어지는 내용이 성공률이 높다.
- 변주(시간이 되면): `TARGET_STYLE="drum_and_bass"`로 한 번 더 실행하면 리듬 밀도 차이가 스펙트로그램에 드러남. `CFG=6`이면 변화와 함께 음량·왜곡도 커짐.

### S24. 실습 B-3 — 전후 비교와 보존 검증 (NB02 셀8·10·12)
- 핵심: 전체 비교(청록 점선 = 편집 경계) → 구간만 확대 비교 → assert로 구간 밖 샘플이 동일함을 확인. 편집 결과에 검증 코드를 함께 둔다.
- 실행: 셀8(≈2초) → 셀10(≈2초) → 셀12 `PASS: 구간 밖 완전 보존, 구간 안 실제 변경.`
- 산출물: `inpaint_full_comparison.html`(오디오 내장, 브라우저로 공유 가능) 열어 보여주기.

### S25. 실습 C — 오디오→오디오 변형 + 수강생 음원
- 핵심: 인페인팅은 구간, 변형은 전체. `noise_level`이 원곡에서 얼마나 멀어질지를 정함. 공식 가이드: 노이즈를 올릴수록 멜로디·리듬 같은 저수준 특징부터 지워지고 음색·조성 같은 고수준 특징만 남는다. 예: 바이올린 솔로 → "Heavy metal guitar" 0.4(음은 유지, 악기만 교체), 트랜스 → 컨트리 0.6(스타일 교체).
- 실행: NB01 셀10 `RUN_VARIATION=True` → `variation(cfg_clips[0][1], "warm jazz trio…", noise_level=0.6, seed=45)` (30초 ≈ 10초). 가능하면 `noise_level=0.3`으로 한 번 더 돌려 A/B.
- 수강생 음원: 미리 받은 30초 이내 파일을 `model/inputs/`에 넣고 NB02 셀3 `SOURCE_PATH=str(MODEL_DIR/"inputs"/"xxx.wav")`, `START/END`를 그 곡 길이 안으로 → 셀4·6·8 재실행. (380초 초과 파일은 거부되므로 사전에 자른다.)

### S26. 실습 D-1 — 실시간 피아노 채보 라이브 (서버 CPU)
- 핵심: 분석 모델은 명령 한 줄과 결과 파일. 30초 클립을 온라인 모드로 돌려 8초 안에 끝나는 것을 보여준다.
- 실행(서버 터미널, Mac에서 ssh 화면 공유):
  ```bash
  cd piano/tools && CUDA_VISIBLE_DEVICES="" ~/anaconda3/envs/piano_rt/bin/python run_piano_amt.py ../songs/scarlatti_sonata_k455_maestro2017.flac --max_seconds 30 --threads 4 --ref_midi ../songs/scarlatti_sonata_k455_maestro2017.midi --out_dir /tmp/class_amt
  ```
  (≈ 8초) → `amt_timing.json`의 `onset_f1`, `end_to_end_latency_estimate_ms` 읽기 → `amt_roll.png` 열기.
- 변주: `--no-delay-compensation` 붙여 다시 → onset F1이 0.99에서 0.2대로 떨어지는 것 확인 (S7의 지연 보정 설명과 연결).
- 백업: `piano/outputs/scarlatti_sonata_k455_maestro2017/amt_timing.json`, `amt_roll.png` 사전 캡처.

### S27. 실습 D-2 — 밴드 분석 JSON 열기 (사전 실행 결과)
- 핵심: Demucs·YourMT3+·allin1은 CPU로 곡당 3~10분이라 수업 중엔 돌리지 않고, 통합 결과 `analysis.json`을 파이썬 몇 줄로 읽어 비주얼용 값을 꺼낸다.
- 실행(서버, 어떤 파이썬이든):
  ```python
  import json; d = json.load(open('band/outputs/02_hello/analysis.json'))
  print(d['tempo']['bpm'], len(d['tempo']['beats']), d['tempo']['downbeats'][:4])
  print([(round(s['start']), s['label']) for s in d['structure']])
  print(d['stems'].keys(), d['frames']['fps'])
  dr = d['stems']['drums']['onset_strength']; print('drum hits >0.5:', sum(x > 0.5 for x in dr))
  ```
  → 출력 예(실측): BPM 128, 비트 476개, 구조 intro→verse→inst→verse→chorus→solo→chorus→outro.
- 자료: 리포트 `#s4`(스템 체크박스)와 `#s6`(클릭 재생)를 옆에 띄워 숫자와 그래프를 대응시킨다.
- 메모: 스템 청취는 `docs/audio/hello_stems/*.mp3`로 바로 가능. `structure_map.png` 한 장으로 마무리.

### S28. 실습 D-3 — 분석 모델 설정값 요약 (명령줄)
- 핵심: 오늘 실행하지 않은 세 모델의 주요 설정값을 한 표로 (3-3, 3-4, 3-5의 설정값 항목). 개인 실행용 참고.
  ```bash
  # Demucs: 보컬/반주 2스템, CPU
  demucs -n htdemucs --two-stems vocals -d cpu song.wav
  # allin1: 단일 모델로 빠르게, 활성화 곡선 저장
  allin1 -d cpu --model harmonix-fold0 --activ song.wav
  # YourMT3+: 서버 래퍼
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 ~/anaconda3/envs/mt3/bin/python band/tools/run_yourmt3.py --device cpu song_dir -o out_dir
  ```

### S29. 정리 — 생성 모델 설정값 요약
- 핵심: S14 표의 "올리면/내리면" 열을 수업 중 청취 결과로 다시 채운다. 수강생이 말한 표현을 그대로 적는다.

### S30. 이후 과제 + 자료 링크
- 핵심: ① Tone Transfer로 녹음 → 악기 변환 ② HF Space `stabilityai/stable-audio-3`로 프롬프트 연습 ③ Apple Silicon이면 배포 ZIP 설치(`bash scripts/setup.sh`) ④ 리포트 HTML을 열어 섹션별 소리와 값 다시 보기.
- 링크: Stable Audio 3 GitHub/HF, Tone Transfer, 리포트 standalone HTML(배포본), 이 계획서.

---

## 6. 사전 준비 체크리스트

### 6-1. 환경 (강사 Mac, 토요일 전날까지)
- [ ] `model/` 폴더가 Mac에 있고 `.conda-env`가 살아 있는지: `bash scripts/start_notebook.sh`로 JupyterLab이 뜨고 커널 목록에 **CTxAI · Stable Audio 3 MLX**가 보이는지.
- [ ] 가중치 4개 검증: `./.conda-env/bin/python scripts/prepare_weights.py --verify-only` (dit_medium_f16 2.9 GB, same_l_encoder/decoder 각 1.7 GB, t5gemma 0.57 GB. 합 6.88 GB).
- [ ] 단위 테스트: `./.conda-env/bin/python -m unittest discover -s tests -v` (4개 통과 확인).
- [ ] 두 노트북을 처음부터 끝까지 1회 실행해 시간 재기 (검증 기록: NB01 43.5초, NB02 24.6초). 그 출력 폴더를 **백업 결과**로 지정해 S20·S23에서 바로 열 수 있게 북마크.
- [ ] `reports/01_generate_music.executed.ipynb`, `02_inpaint_music.executed.ipynb`(오디오 내장)도 JupyterLab에서 열리는지 확인 — 모델이 안 돌 때 최후 백업.
- [ ] 저장 공간 15 GB 이상, 전원 연결, 다른 무거운 앱 종료(통합 메모리 18 GB 기준 피크 3.9 GiB).
- [ ] 오프라인 모드(`HF_HUB_OFFLINE=1`)가 스크립트에 이미 설정되어 있으므로 강의장 인터넷 불필요. Tone Transfer만 인터넷 필요.
- [ ] 이 서버의 `model.zip`(7.0 GB, 가중치·conda 환경 포함)은 Linux에서 실행 불가(MLX). 서버에는 참고용으로만 둔다.

### 6-2. 분석 모델 (연구실 Linux 서버, CPU만)
- [ ] `piano_rt` env로 S26 명령을 미리 1회 실행해 `/tmp/class_amt`가 생기는지, 8초 내 끝나는지 확인. (`CUDA_VISIBLE_DEVICES=""` 고정.)
- [ ] `band/outputs/02_hello/analysis.json`, `structure_map.png`, `docs/audio/hello_stems/*.mp3` 존재 확인(확인됨, 2026-10-19).
- [ ] 강의장에서 서버 접속이 불안하면 S26·S27을 사전 캡처(터미널 출력 + PNG)로 대체.

### 6-3. 리포트 HTML
- [ ] `docs/report_2026-10-06_standalone.html`(단일 파일)을 강사 Mac에 복사해 브라우저 북마크: `#s1, #s3, #s4, #s5, #s6, #s7`. 섹션 2(CHIME)는 스크롤로 지나치되 언급하지 않음.
- [ ] 각 섹션 ▶ 재생 버튼과 스템 체크박스(`#s4`)가 오프라인에서도 동작하는지 확인(오디오가 base64로 내장된 standalone 버전 권장).

### 6-4. 샘플 곡
| 용도 | 파일 | 비고 |
|---|---|---|
| 인페인팅 기본 | 노트북이 생성(24초 피아노, seed 72) | 외부 파일 불필요 |
| 인페인팅 실곡 예시 | `docs/audio/hello_60s.mp3` → `model/inputs/hello_30s.wav`로 30초 자르기 (`ffmpeg -i hello_60s.mp3 -t 30 -ar 44100 -ac 2 hello_30s.wav`) | 밴드 곡 20~32초를 오케스트라로 |
| 변형 | NB01 CFG 1 결과(자동) | — |
| 피아노 채보 라이브 | `piano/songs/scarlatti_sonata_k455_maestro2017.flac` + `.midi` | 정답 MIDI가 있어 F1이 바로 나옴 |
| 수강생 음원 | 금요일까지 30초 이내 wav/mp3 수집 → `model/inputs/` | 380초 초과·저작권 민감 파일 제외 |

### 6-5. 수강생용 배포
- [ ] `model/dist/CTxAI-Stable-Audio-3-Medium-MLX.zip`(코드·노트북·스크립트, 가중치 제외) 공유 링크. Apple Silicon + macOS 15 + 15 GB + 인터넷 필요. 설치: `bash scripts/setup.sh` → `bash scripts/start_notebook.sh`.
- [ ] 비(非) Mac 수강생용 링크: HF Space `stabilityai/stable-audio-3`, Tone Transfer, 리포트 standalone HTML.

---

## 7. 리포트 HTML 섹션 → 슬라이드 매핑

| 리포트 섹션 (앵커) | 쓰는 슬라이드 | 쓰는 방식 |
|---|---|---|
| 상단 흐름도 (Input→Models→Output→Visual mapping) | S1, S17 | 캡처 1장. 수업 전체 지도 |
| 1. 실시간 피아노 채보 `#s1` (canvas `c1`, ▶ 재생 1) | S2, S7, S26 | 10초 재생 → S7에서 숫자 카드(F1, 지연, 배속, fps) → 토요일 라이브 결과와 대조 |
| 2. 멜로디 추출 (CHIME) `#s2` | 사용 안 함 | 수업 범위 밖 |
| 3. 프레임 요약값 `#s3` (canvas `c3`) | S8, S27 | 곡별 밀도·높이·onset 빈도 표 + 비주얼 매핑 예. 토요일에는 요약값 계산 예로 재인용 |
| 4. 스템 분리 `#s4` (canvas `c4`, 스템 체크박스) | S9, S27 | 보컬/드럼만 켜고 청취. RMS envelope = 비주얼 레이어 밝기 |
| 5. 멀티악기 채보 `#s5` (canvas `c5`, 곡 표) | S10 | 5묶음 피아노롤 + 드럼 onset_strength. 곡 표에서 실제로 없는 악기 라벨 지적 |
| 6. 비트·구조 `#s6` (canvas `c6`, 클릭 재생) | S11, S27 | 클릭과 구간 띠 재생. BPM 표와 〈다시, 다시 만나〉 편차 97 ms 사례 |
| 7. 제공 가능 값 목록 `#s7` | S6(간략), S29 | 신뢰도 열 강조: 후처리가 필요한 값 구분 |
| 하단 CPU 실행 노트 (RTF 수치) | S4, S28 | 추론은 CPU로 충분하다는 근거 |

---

## 8. 리스크와 대안

| 상황 | 대안 |
|---|---|
| Mac에서 `No Metal device available` | 원격 세션·샌드박스가 아닌 로컬 터미널에서 실행. 안 되면 `reports/*.executed.ipynb`로 진행 |
| 추론이 예상보다 느림(다른 Mac) | `SECONDS=15`, `CFG_VALUES=[1,4]`로 줄이면 A 블록이 약 1/4로 단축 |
| 메모리 부족 | 길이 줄이기, CFG 1, 한 번에 하나. `--free-models`는 기본 켜짐 |
| 서버 접속 불가 | S26·S27은 사전 캡처 + `piano/outputs/*`, `band/outputs/*` 복사본을 Mac에 가져가 열기 |
| Tone Transfer 사이트 변경/접속 불가 | 사전 녹화 30초 영상 1개 준비 |
| 수강생 음원이 길거나 포맷 문제 | ffmpeg로 30초 자르기 명령을 S25에 준비. 변환 실패 시 노트북 기본 피아노 예제로 진행 |
| 시간 초과 | 4절·1절의 자르는 순서대로 |

## 9. 참고 링크
- Stable Audio 3: https://github.com/Stability-AI/stable-audio-3 · 가중치(최적화판) https://huggingface.co/stabilityai/stable-audio-3-optimized · 데모 https://huggingface.co/spaces/stabilityai/stable-audio-3 · 기술 보고서 arXiv:2605.17991
- Tone Transfer: https://magenta.withgoogle.com/tone-transfer (앱: https://sites.research.google/tonetransfer)
- online_amt: https://github.com/jdasam/online_amt · Demucs: https://github.com/facebookresearch/demucs · YourMT3+: https://huggingface.co/spaces/mimbres/YourMT3 · allin1: https://github.com/mir-aidj/all-in-one
- 프로젝트 문서: `docs/value_catalog.md`(값 설명), `docs/piano_models_setup.md`, `docs/yourmt3_setup.md`, `docs/allin1_setup.md`, 리포트 `docs/report_2026-10-06.html`
