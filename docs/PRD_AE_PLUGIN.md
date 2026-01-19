# Everyric2 After Effects Plugin PRD

> Product Requirements Document v1.0
> Last Updated: 2025-01-19

---

## 1. 개요

### 1.1 제품 비전
After Effects에서 **원클릭으로** 오디오에 싱크된 가사 자막을 생성하는 플러그인.
영상 편집자가 가사 타이밍을 수동으로 맞추는 시간을 **90% 이상 절감**.

### 1.2 킬링 포인트
| 기존 워크플로우 | Everyric2 |
|----------------|-----------|
| 수동 타이밍 (30분~2시간) | **자동 생성 (~30초)** |
| LRC 파일 찾기/변환 | **자동 검색 + 생성** |
| 외부 도구로 왔다갔다 | **AE 내에서 완결** |
| 타이밍 수정 번거로움 | **비주얼 에디터** |

### 1.3 타겟 사용자
- 뮤직비디오 편집자
- 리릭 비디오 크리에이터
- 유튜브/TikTok 쇼츠 제작자
- 자막 작업자
- 커버/인디 아티스트

---

## 2. 아키텍처

### 2.1 하이브리드 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                    After Effects Plugin (CEP Panel)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [모드 선택]                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │  🖥️ 로컬 모드        │  │  ☁️ 클라우드 모드    │                  │
│  │  (everyric2 CLI)    │  │  (API 서버)         │                  │
│  │  무료, 직접 설치     │  │  간편, 후원/유료     │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│                                                                     │
│  [워크플로우]                                                        │
│  1. 오디오 레이어 선택                                               │
│  2. 가사 입력/검색/붙여넣기                                          │
│  3. "생성" 클릭                                                      │
│  4. 타임라인에 마커/자막 레이어 자동 생성                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  로컬 모드               │  │  클라우드 모드           │
├─────────────────────────┤  ├─────────────────────────┤
│  everyric2 CLI 호출     │  │  Everyric API 호출      │
│  - 오디오 파일 직접 처리  │  │  - 오디오 업로드         │
│  - GPU 사용 (있으면)     │  │  - 서버에서 처리         │
│  - 무료                  │  │  - 결과 다운로드         │
│  - 설치 필요             │  │  - 간편, 후원/유료       │
└─────────────────────────┘  └─────────────────────────┘
```

### 2.2 기술 스택

```
CEP Panel (HTML/CSS/JS)
    │
    ├─ Node.js (CEP 환경)
    │   ├─ 파일 시스템 접근
    │   ├─ 프로세스 실행 (everyric2 CLI)
    │   └─ HTTP 요청 (클라우드 API)
    │
    └─ ExtendScript (JSX)
        ├─ AE 컴포지션 조작
        ├─ 마커 생성
        ├─ 텍스트 레이어 생성
        └─ 키프레임 설정
```

---

## 3. 기능 명세

### 3.1 Core 기능

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| 싱크 생성 | 오디오 + 가사 → 타임스탬프 | P0 |
| 마커 생성 | 타임라인에 가사별 마커 | P0 |
| 자막 레이어 생성 | 텍스트 레이어 자동 생성 | P0 |
| 가사 검색 | LRCLIB에서 자동 검색 | P1 |
| 로컬/클라우드 선택 | 처리 모드 선택 | P0 |
| LRC 내보내기 | 결과를 LRC 파일로 저장 | P1 |

### 3.2 출력 옵션

| 출력 타입 | 설명 | 사용 사례 |
|----------|------|----------|
| **마커** | 컴포지션 타임라인에 마커 | 참조용, 수동 편집 |
| **텍스트 레이어** | 가사별 텍스트 레이어 생성 | 리릭 비디오 |
| **Essential Graphics** | 템플릿 연동 | Premiere 연동 |
| **LRC 파일** | 타임스탬프 파일 저장 | 외부 사용 |
| **SRT/ASS 파일** | 자막 파일 저장 | 자막 용도 |

### 3.3 가사 입력 방식

| 방식 | 설명 | 우선순위 |
|------|------|----------|
| 직접 입력 | 텍스트 박스에 붙여넣기 | P0 |
| LRCLIB 검색 | 아티스트/제목으로 검색 | P1 |
| LRC 파일 불러오기 | 기존 LRC 파일 임포트 | P1 |
| 클립보드 | 클립보드에서 자동 감지 | P2 |

### 3.4 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| 처리 모드 | 클라우드 | 로컬/클라우드 |
| everyric2 경로 | 자동 감지 | 로컬 모드용 CLI 경로 |
| 출력 형식 | 마커 | 마커/텍스트 레이어/둘 다 |
| 언어 | 자동 감지 | 가사 언어 힌트 |
| 마커 색상 | 초록 | 생성된 마커 색상 |

---

## 4. UI/UX

### 4.1 메인 패널

```
┌──────────────────────────────────────────┐
│  🎵 Everyric2                    [⚙️]    │
├──────────────────────────────────────────┤
│                                          │
│  오디오 소스                              │
│  ┌────────────────────────────────────┐ │
│  │ [선택된 레이어 없음]         [선택] │ │
│  └────────────────────────────────────┘ │
│                                          │
│  가사                                    │
│  ┌────────────────────────────────────┐ │
│  │                                    │ │
│  │  여기에 가사를 붙여넣거나           │ │
│  │  검색하세요...                      │ │
│  │                                    │ │
│  │                                    │ │
│  └────────────────────────────────────┘ │
│  [🔍 LRCLIB 검색] [📁 LRC 불러오기]     │
│                                          │
│  출력 형식                               │
│  ○ 마커만  ○ 텍스트 레이어  ● 둘 다     │
│                                          │
│  처리 모드                               │
│  ○ 로컬 (everyric2)  ● 클라우드         │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │         ✨ 싱크 생성하기            │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ────────────────────────────────────── │
│  상태: 준비됨                            │
└──────────────────────────────────────────┘
```

### 4.2 처리 중 상태

```
┌──────────────────────────────────────────┐
│  🎵 Everyric2                    [⚙️]    │
├──────────────────────────────────────────┤
│                                          │
│  ⏳ 싱크 생성 중...                       │
│                                          │
│  [■■■■■■■□□□] 70%                        │
│                                          │
│  현재: 오디오 정렬 중                     │
│  예상 시간: ~10초                         │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │            ❌ 취소                  │ │
│  └────────────────────────────────────┘ │
│                                          │
└──────────────────────────────────────────┘
```

### 4.3 완료 상태

```
┌──────────────────────────────────────────┐
│  🎵 Everyric2                    [⚙️]    │
├──────────────────────────────────────────┤
│                                          │
│  ✅ 싱크 생성 완료!                       │
│                                          │
│  📍 마커 42개 생성됨                      │
│  📝 텍스트 레이어 42개 생성됨             │
│                                          │
│  [LRC 저장] [SRT 저장] [다시 생성]        │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │         🔄 새로운 작업              │ │
│  └────────────────────────────────────┘ │
│                                          │
└──────────────────────────────────────────┘
```

### 4.4 LRCLIB 검색 모달

```
┌──────────────────────────────────────────┐
│  🔍 LRCLIB에서 가사 검색                  │
├──────────────────────────────────────────┤
│                                          │
│  검색어                                  │
│  ┌────────────────────────────────────┐ │
│  │ Never Gonna Give You Up            │ │
│  └────────────────────────────────────┘ │
│                                          │
│  검색 결과:                              │
│  ┌────────────────────────────────────┐ │
│  │ ● Rick Astley - Never Gonna...    │ │
│  │   ♪ 싱크됨 | 3:32                  │ │
│  │ ○ Various - Never Gonna Give...   │ │
│  │   ♪ 플레인 | 3:30                  │ │
│  └────────────────────────────────────┘ │
│                                          │
│  [취소]                    [선택]        │
│                                          │
└──────────────────────────────────────────┘
```

---

## 5. 기술 스펙

### 5.1 CEP Panel 구조

```
everyric2-ae/
├── CSXS/
│   └── manifest.xml          # CEP 매니페스트
├── js/
│   ├── main.js               # 메인 UI 로직
│   ├── api.js                # API 클라이언트
│   ├── local.js              # 로컬 CLI 호출
│   └── lrclib.js             # LRCLIB 연동
├── jsx/
│   └── host.jsx              # ExtendScript (AE 조작)
├── css/
│   └── style.css             # 스타일
├── index.html                # 메인 HTML
└── .debug                    # 개발 모드 설정
```

### 5.2 manifest.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ExtensionManifest Version="7.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <ExtensionList>
    <Extension Id="com.everyric.ae" Version="1.0.0"/>
  </ExtensionList>
  <ExecutionEnvironment>
    <HostList>
      <Host Name="AEFT" Version="[23.0,99.9]"/>  <!-- AE 2023+ -->
    </HostList>
    <LocaleList>
      <Locale Code="All"/>
    </LocaleList>
    <RequiredRuntimeList>
      <RequiredRuntime Name="CSXS" Version="11.0"/>
    </RequiredRuntimeList>
  </ExecutionEnvironment>
  <DispatchInfoList>
    <Extension Id="com.everyric.ae">
      <DispatchInfo>
        <Resources>
          <MainPath>./index.html</MainPath>
          <CEFCommandLine>
            <Parameter>--enable-nodejs</Parameter>
          </CEFCommandLine>
        </Resources>
        <UI>
          <Type>Panel</Type>
          <Menu>Everyric2</Menu>
          <Geometry>
            <Size>
              <Width>300</Width>
              <Height>500</Height>
            </Size>
          </Geometry>
        </UI>
      </DispatchInfo>
    </Extension>
  </DispatchInfoList>
</ExtensionManifest>
```

### 5.3 ExtendScript (JSX) - 마커 생성

```javascript
// host.jsx

function createMarkersFromTimestamps(timestampsJson) {
  var timestamps = JSON.parse(timestampsJson);
  var comp = app.project.activeItem;
  
  if (!(comp instanceof CompItem)) {
    return JSON.stringify({ error: "No active composition" });
  }
  
  // 기존 마커 삭제 옵션
  // comp.markerProperty.removeMarker(...)
  
  var created = 0;
  for (var i = 0; i < timestamps.length; i++) {
    var ts = timestamps[i];
    var marker = new MarkerValue(ts.text);
    marker.duration = ts.end - ts.start;
    
    comp.markerProperty.setValueAtTime(ts.start, marker);
    created++;
  }
  
  return JSON.stringify({ success: true, created: created });
}

function createTextLayersFromTimestamps(timestampsJson, options) {
  var timestamps = JSON.parse(timestampsJson);
  var opts = JSON.parse(options);
  var comp = app.project.activeItem;
  
  if (!(comp instanceof CompItem)) {
    return JSON.stringify({ error: "No active composition" });
  }
  
  app.beginUndoGroup("Everyric2 - Create Text Layers");
  
  var created = 0;
  for (var i = 0; i < timestamps.length; i++) {
    var ts = timestamps[i];
    
    // 텍스트 레이어 생성
    var textLayer = comp.layers.addText(ts.text);
    textLayer.startTime = ts.start;
    textLayer.outPoint = ts.end;
    
    // 기본 스타일 적용
    var textProp = textLayer.property("Source Text");
    var textDoc = textProp.value;
    textDoc.fontSize = opts.fontSize || 60;
    textDoc.font = opts.font || "Arial";
    textDoc.fillColor = [1, 1, 1]; // 흰색
    textDoc.justification = ParagraphJustification.CENTER_JUSTIFY;
    textProp.setValue(textDoc);
    
    // 위치 설정
    var position = textLayer.property("Position");
    position.setValue([comp.width / 2, comp.height * 0.85]);
    
    created++;
  }
  
  app.endUndoGroup();
  
  return JSON.stringify({ success: true, created: created });
}
```

### 5.4 Node.js - 로컬 CLI 호출

```javascript
// local.js

const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

async function runLocalAlignment(audioPath, lyrics, options = {}) {
  // 임시 파일에 가사 저장
  const lyricsPath = path.join(os.tmpdir(), `everyric_${Date.now()}.txt`);
  fs.writeFileSync(lyricsPath, lyrics);
  
  // everyric2 CLI 호출
  const cliPath = options.cliPath || 'everyric2';
  const outputPath = path.join(os.tmpdir(), `everyric_${Date.now()}.json`);
  
  const cmd = `${cliPath} align "${audioPath}" "${lyricsPath}" --output "${outputPath}" --format json`;
  
  return new Promise((resolve, reject) => {
    exec(cmd, { timeout: 120000 }, (error, stdout, stderr) => {
      // 임시 파일 정리
      fs.unlinkSync(lyricsPath);
      
      if (error) {
        reject(new Error(`CLI error: ${stderr}`));
        return;
      }
      
      const result = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
      fs.unlinkSync(outputPath);
      
      resolve(result);
    });
  });
}

module.exports = { runLocalAlignment };
```

### 5.5 API 클라이언트

```javascript
// api.js

const API_BASE = 'https://api.everyric.com';

async function generateSyncCloud(audioFile, lyrics, options = {}) {
  // 1. 오디오 업로드
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('lyrics', lyrics);
  formData.append('language', options.language || 'auto');
  
  const response = await fetch(`${API_BASE}/api/align`, {
    method: 'POST',
    body: formData,
    headers: {
      'Authorization': `Bearer ${options.apiKey || ''}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  const result = await response.json();
  
  // 2. 폴링 (작업 완료 대기)
  if (result.job_id) {
    return await pollJobStatus(result.job_id);
  }
  
  return result;
}

async function pollJobStatus(jobId, maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    const response = await fetch(`${API_BASE}/api/job/${jobId}`);
    const status = await response.json();
    
    if (status.status === 'completed') {
      return status;
    } else if (status.status === 'failed') {
      throw new Error(status.error);
    }
    
    await new Promise(r => setTimeout(r, 2000)); // 2초 대기
  }
  
  throw new Error('Timeout waiting for job completion');
}

module.exports = { generateSyncCloud };
```

---

## 6. 배포 전략

### 6.1 오픈소스 + 빌드 유료 (Aseprite 모델)

```
┌─────────────────────────────────────────────────────────────────────┐
│  배포 옵션                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [무료] GitHub에서 직접 빌드                                         │
│  - 코드 100% 공개                                                   │
│  - 직접 빌드 필요 (개발자용)                                         │
│  - everyric2 CLI 별도 설치 필요                                     │
│                                                                     │
│  [유료*] 빌드된 ZXP 다운로드                                         │
│  - 원클릭 설치 (.zxp)                                               │
│  - 가이드 문서 포함                                                  │
│  - * MMS가 CC-BY-NC라서 현재는 후원/무료                             │
│                                                                     │
│  [클라우드] API 크레딧                                               │
│  - 설치 없이 사용                                                   │
│  - 후원 시 크레딧 제공                                               │
│  - 서버 비용 충당                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 현재 (CC-BY-NC 제약)

| 옵션 | 가격 | 대상 |
|------|------|------|
| GitHub 빌드 | 무료 | 개발자 |
| ZXP 다운로드 | 무료 (후원 환영) | 일반 사용자 |
| 클라우드 API | 후원 기반 | 편의 추구 |

### 6.3 미래 (상업화 시)

| 옵션 | 가격 | 대상 |
|------|------|------|
| GitHub 빌드 | 무료 | 개발자 |
| ZXP 다운로드 | $30-50 | 일반 사용자 |
| 클라우드 API | $5/50곡 | 가끔 사용 |
| 클라우드 구독 | $10/월 무제한 | 헤비 사용자 |

---

## 7. 로드맵

### Phase 1: MVP (4주)
- [ ] CEP Panel 기본 구조
- [ ] 로컬 모드 (everyric2 CLI 호출)
- [ ] 마커 생성 기능
- [ ] 텍스트 레이어 생성 기능
- [ ] 기본 UI

### Phase 2: 클라우드 연동 (3주)
- [ ] 클라우드 API 연동
- [ ] 오디오 업로드 + 처리
- [ ] 진행 상태 표시
- [ ] 후원/크레딧 시스템

### Phase 3: 고급 기능 (4주)
- [ ] LRCLIB 검색 연동
- [ ] 다양한 출력 형식 (SRT, ASS)
- [ ] Essential Graphics 템플릿
- [ ] 스타일 프리셋
- [ ] 키프레임 애니메이션 옵션

### Phase 4: 확장 (4주)
- [ ] Premiere Pro 지원
- [ ] DaVinci Resolve 지원 (검토)
- [ ] 배치 처리
- [ ] 프로젝트 템플릿

---

## 8. 성공 지표

| 지표 | 목표 (3개월) | 목표 (6개월) |
|------|-------------|-------------|
| GitHub Stars | 100 | 500 |
| 다운로드 | 500 | 2,000 |
| MAU | 50 | 200 |
| 클라우드 사용량 | 200곡/월 | 1,000곡/월 |
| 후원자 | 10명 | 50명 |

---

## 9. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| CEP 지원 종료 | 중 | 높음 | UXP 마이그레이션 준비 |
| AE 버전 호환성 | 높음 | 중 | 버전별 테스트, 최소 버전 명시 |
| 서버 비용 증가 | 중 | 중 | Rate limit, 후원 강화 |
| CC-BY-NC 제약 | 현재 | 중 | MFA 대안 준비, 상업화 시 교체 |

---

## 10. 경쟁 분석

| 도구 | 가격 | 자동 싱크 | AE 연동 | 단점 |
|------|------|----------|---------|------|
| **Everyric2** | 무료/후원 | ✅ AI | ✅ 네이티브 | 신규 |
| AEJuice Lyrics | $39 | ❌ 수동 | ✅ 플러그인 | 수동 타이밍 |
| 수동 작업 | 무료 | ❌ | - | 시간 소요 |
| LRC 변환 스크립트 | 무료 | ❌ | ⚠️ 스크립트 | LRC 필요 |

---

## 부록: everyric2 CLI 연동 스펙

### CLI 설치 확인
```javascript
async function checkCliInstalled() {
  try {
    const { stdout } = await execPromise('everyric2 --version');
    return { installed: true, version: stdout.trim() };
  } catch {
    return { installed: false, version: null };
  }
}
```

### CLI 호출 형식
```bash
everyric2 align <audio_file> <lyrics_file> \
  --output <output_path> \
  --format json \
  --engine ctc \
  --language auto
```

### 출력 JSON 형식
```json
{
  "segments": [
    {
      "text": "Never gonna give you up",
      "start": 0.0,
      "end": 2.5,
      "confidence": 0.95,
      "words": [
        { "word": "Never", "start": 0.0, "end": 0.5 },
        { "word": "gonna", "start": 0.5, "end": 0.9 }
      ]
    }
  ],
  "metadata": {
    "duration": 213.5,
    "language": "en",
    "engine": "ctc"
  }
}
```
