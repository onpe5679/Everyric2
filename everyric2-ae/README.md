# Everyric2 After Effects Plugin

After Effects에서 원클릭으로 오디오에 싱크된 가사 자막을 생성하는 CEP 플러그인.

## 기능

- **자동 가사 싱크**: everyric2 CLI 또는 클라우드 API를 통한 자동 타이밍
- **마커 생성**: 컴포지션 타임라인에 가사별 마커 추가
- **텍스트 레이어 생성**: 자동으로 타이밍된 텍스트 레이어 생성
- **LRCLIB 검색**: 온라인에서 가사 검색
- **번역 & 발음**: 한국어 번역 및 로마자 발음 표기 지원

## 설치

### 요구사항

- After Effects 2023 (v23.0) 이상
- Node.js 18+
- everyric2 CLI (로컬 모드 사용 시)

### 빌드 및 설치

```bash
cd everyric2-ae
npm install
npm run build
npm run install-plugin
```

### Debug 모드 활성화 (개발용)

**Windows:**
1. 레지스트리 편집기 (regedit) 실행
2. `HKEY_CURRENT_USER\SOFTWARE\Adobe\CSXS.11` 이동
3. 문자열 값 `PlayerDebugMode` = `1` 추가

**macOS:**
```bash
defaults write com.adobe.CSXS.11 PlayerDebugMode 1
```

### After Effects에서 열기

1. After Effects 재시작
2. `Window > Extensions > Everyric2`

## 사용법

1. 오디오가 포함된 컴포지션 열기
2. "Select" 버튼으로 오디오 레이어 선택
3. 가사 입력 (직접 붙여넣기 또는 LRCLIB 검색)
4. 출력 형식 선택 (Markers / Text Layers / Both)
5. "Generate Sync" 클릭

## 개발

```bash
# 의존성 설치
npm install

# 빌드
npm run build

# 감시 모드 (패널만)
npm run watch:panel

# 플러그인 설치
npm run install-plugin
```

## 프로젝트 구조

```
everyric2-ae/
├── CSXS/
│   └── manifest.xml      # CEP 매니페스트
├── src/
│   ├── panel/            # 패널 TypeScript 소스
│   │   ├── main.ts       # 메인 UI 로직
│   │   ├── local.ts      # 로컬 CLI 호출
│   │   ├── api.ts        # 클라우드 API 클라이언트
│   │   ├── lrclib.ts     # LRCLIB 연동
│   │   └── types.ts      # 타입 정의
│   └── types/            # 타입 정의
│       ├── cep.d.ts      # CEP 인터페이스
│       └── extendscript.d.ts
├── dist/
│   ├── js/main.js        # 빌드된 패널 JS
│   └── jsx/host.jsx      # ExtendScript (AE 조작)
├── css/style.css         # 스타일
├── index.html            # 메인 HTML
├── .debug                # 디버그 설정
└── scripts/install.js    # 설치 스크립트
```

## 처리 모드

### 로컬 모드 (권장)
- everyric2 CLI 직접 호출
- GPU 가속 지원
- 무료

### 클라우드 모드
- API 서버 사용
- 설치 불필요
- 인터넷 연결 필요

## 라이선스

MIT License
