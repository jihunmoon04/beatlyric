# Lyric Study Video Pipeline

좋아하는 음악(음원/영상)에서 **보컬 분리 · 가사 강제 정렬 · 박자 악보 양자화(`0.5`박/`0.25`박) · 발음/번역/학습 코멘트**를 생성한 뒤, **Beat Score Studio** 웹 UI에서 실시간 미리보기로 검수·조절하고 즉시 1080p MP4 학습 영상으로 렌더링하는 파이프라인입니다.

---

## 1. 전체 파이프라인 아키텍처 & LLM 개입 지점

```text
[Step 1] 음원/에셋 추출 (step1_ingest.py)
   └─ master_audio.m4a, analysis.wav, cover.jpg, bg_blur.png, meta.json
        │
[Step 2] 보컬 분리 & 강제 정렬 (step2_align.py)
   └─ LRC/TXT 로드 + Demucs(vocals.wav) + ctc-forced-aligner / stable-ts
        │
[Step 3] 오디오 스펙트럼 추출 (step3_spectrum.py)
   └─ spectrum.json (30fps, 28 bars)
        │
[Step 4] 박자 검출 & 보컬 온셋 기반 악보 양자화 (step4_score.py)
   └─ analysis.wav BPM/t0 추정 + vocals.wav 에너지 온셋 대조 0.5박/0.25박 스냅
      → score.json / score_initial.json / timeline.json
        │
[Step 5] ★ LLM 개입 단계 ★ (step5_enrich.py)
   └─ 타이밍/박자/단어 좌표는 100% 보존하고 언어 학습 정보만 생성·병합
      (pronunciation 한글 발음, translation 번역, comment 학습 노트)
      • 방식 A: Gemini API 자동 호출 (GEMINI_API_KEY / --api-key)
      • 방식 B: AI 에이전트(채팅창)에 llm_prompt.md 전달 후 --from-json 병합
        │
[Step 6] Beat Score Studio 검수 & 미세 조정 (studio_server.py / studio_ui.html)
   └─ 고정 16:9 1920x1080 실시간 프리뷰 · 0.5박 줄 이동/길이 · 0.25박 단어 이동/길이
      우클릭 드래그 패닝 · 블록 우클릭 한줄전체(whole_line) 전환 · Undo/초기화
        │
[Step 7] 확인 완료 · MP4 렌더링 (render_lib.py + Remotion + FFmpeg)
   └─ 스튜디오에서 [확인 완료 · MP4 렌더링] 클릭 → output/<곡>_<스타일>.mp4 생성
```

> **LLM이 개입하는 곳:** 오직 **Step 5 (`step5_enrich.py`)** 한 곳입니다.  
> 박자와 가사 타이밍은 오디오(Demucs + CTC/Whisper + `vocals.wav` 온셋)와 사용자가 Studio에서 맞춘 `score.json`을 기준으로 결정되며, LLM은 **가사별 발음(`pronunciation`), 의역(`translation`), 표현 해설(`comment`)**만 채워 넣습니다.

---

## 2. 실행 방법

### A. 원클릭 통합 실행 (`auto`)
음원 분석부터 박자 악보 생성, LLM 주석(선택), 웹 스튜디오 자동 실행까지 한 번에 수행합니다.

```bash
# 1) Gemini API 키가 있을 때 (주석까지 완전 자동 생성 후 스튜디오 오픈)
uv run python pipeline/run.py auto input/my_song.mp3 --lyrics input/lyrics.txt --lang en --api-key YOUR_KEY

# 2) API 키 없이 에이전트/수동 주석 JSON과 함께 실행할 때
uv run python pipeline/run.py auto input/my_song.mp3 --lyrics input/lyrics.txt --annotations annotations.json --lang en

# 3) 주석 없이 먼저 박자 검수 스튜디오부터 열 때 (work/<곡>/llm_prompt.md 자동 생성)
uv run python pipeline/run.py auto input/my_song.mp3 --lyrics input/lyrics.txt --lang en
```

### B. 단계별 실행 (`prepare` → `enrich` → `edit` → `render`)

```bash
# Step 1~4: 에셋 추출 + 보컬 정렬 + 스펙트럼 + 보컬 온셋 0.5박 score.json 생성
uv run python pipeline/run.py prepare input/my_song.mp3 --lyrics input/lyrics.txt --lang en --force-align

# Step 5: LLM 발음·번역·코멘트 병합 (Gemini API 또는 에이전트 JSON)
uv run python pipeline/run.py enrich my_song --api-key YOUR_KEY
# 또는
uv run python pipeline/run.py enrich my_song --from-json work/my_song/annotations.json

# Step 6: Beat Score Studio 실행 (http://127.0.0.1:8765)
uv run python pipeline/run.py edit work/my_song --style minimal --output output/my_song_minimal.mp4

# Step 7: CLI로 직접 MP4 렌더링 (스튜디오의 렌더링 버튼과 동일)
uv run python pipeline/run.py render work/my_song --style minimal
```

---

## 3. Beat Score Studio 주요 조작법

- **프리뷰 비율 고정 & 배율 조절**: 내부를 고정 `1920x1080` 캔버스로 렌더링해 창 크기에 따라 구도가 망가지지 않으며, `미리보기 창 크기`(`작게/기본/크게/전체너비`)와 `영상 내부 비율`(`content_scale`, 기본 `118%`)을 조절해 MP4에 그대로 반영할 수 있습니다.
- **가사 줄 이동 / 길이 조절**: 전체 박자 악보 블록을 좌드래그해 `0.5`박 단위로 이동하고, 우측 끝 핸들로 길이를 조절합니다. (`[ / ]` 키, `T` 키로 현재 재생 위치 스냅 지원)
- **줄 내부 단어 박자 & 길이 조절**: 하단 단어 레인에서 단어 앞 눈금을 드래그해 `0.25`박 단위로 시작점을 옮기고, 단어 블록 우측 끝 핸들 또는 상단 `길이 ±0.25` 버튼으로 길이를 조절합니다.
- **노란색 재생 바 스크럽**: 단어 레인 빈 공간이나 눈금을 클릭·드래그하면 해당 구간의 오디오 위치로 즉시 이동합니다.
- **우클릭 조작**:
  - **우클릭 드래그**: 전체 박자 악보 및 단어 레인을 좌우로 부드럽게 스크롤합니다.
  - **가사 블록 우클릭(클릭)**: 단어 순차 강조를 끄고 줄 전체가 한 번에 표시되는 **`한줄전체(whole_line)`** 모드를 토글합니다.
- **히스토리 & 초기화**: 상단 `↶ 뒤로가기`(`Ctrl+Z`), `↷ 앞으로 가기`(`Ctrl+Y`), `알고리즘 박자로 초기화` 버튼으로 언제든 되돌릴 수 있습니다.

---

## 4. 설치 안내 (`uv` + Node.js)

```bash
# Python 패키지 (기본 + 정렬 의존성)
uv sync --extra align

# Remotion 렌더러 의존성
cd renderer
npm install
npx remotion browser ensure
```

### 디자인 프리셋 (`--style`)
1. `minimal` (**기본/추천**): 모던 블러 배경 + 앨범 아트 아래 대칭 미니 웨이브 + 포커스 가사 스크롤
2. `editorial`: 여백이 넓은 2단 에디토리얼 레이아웃 + LED 스펙트럼 + 왼쪽 포인트 바 코멘트
3. `classic2`: 세리프 타이포그래피 + LED 스펙트럼 + 하이라이트 보더 중심 레이아웃
