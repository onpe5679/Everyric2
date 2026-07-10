// 번역/헬스 엔드포인트 목업 — 데모·스모크 테스트용 (실서버 없이 번역 UI 확인)
// 실행: node scripts/mock-server.mjs
// EVERYRIC_MOCK_SYNC=<timestamps.json> 지정 시 /api/sync/*가 해당 싱크를 반환
// (notes 포함 실측 데이터를 물려 음정 바 E2E에 사용)
import http from 'http';
import { readFileSync } from 'fs';

let mockSync = null;
if (process.env.EVERYRIC_MOCK_SYNC) {
  try {
    mockSync = JSON.parse(readFileSync(process.env.EVERYRIC_MOCK_SYNC, 'utf8'));
    console.log(`mock sync loaded: ${mockSync.length} segments`);
  } catch (e) {
    console.error('mock sync load failed:', String(e).slice(0, 200));
  }
}

const DICT = [
  ['no strangers to love', '우린 사랑이 처음은 아니잖아'],
  ['know the rules and so do i', '너도 나도 규칙은 알고 있지'],
  ["full commitment's what i", '내가 생각하는 건 완전한 헌신이야'],
  ["wouldn't get this from any other guy", '다른 남자에게선 얻지 못할 거야'],
  ['wanna tell you how i', '내 마음이 어떤지 그저 말하고 싶어'],
  ['make you understand', '네가 알아줬으면 해'],
  ['never gonna give you up', '절대 널 포기하지 않아'],
  ['never gonna let you down', '절대 널 실망시키지 않아'],
  ['never gonna run around', '절대 한눈팔거나 널 떠나지 않아'],
  ['never gonna make you cry', '절대 널 울리지 않아'],
  ['never gonna say goodbye', '절대 작별 인사는 하지 않아'],
  ['never gonna tell a lie', '절대 거짓말로 널 아프게 하지 않아'],
  ['known each other for so long', '우린 오랫동안 서로를 알아왔지'],
  ["heart's been aching", '네 마음은 아파도 수줍어 말 못 하지'],
  ["both know what's been going on", '속으론 우리 둘 다 무슨 일인지 알아'],
  ['know the game and', '우린 이 게임을 알고, 함께할 거야'],
  ['ask me how i', '내 기분이 어떤지 묻는다면'],
  ['too blind to see', '모른 척하지는 말아줘'],
  ['give you up', '(널 포기해)'],
  ['give, never gonna give', '(절대, 절대 포기 안 해)'],
];

function translateLine(line) {
  const lower = line.toLowerCase();
  for (const [key, tr] of DICT) {
    if (lower.includes(key)) return tr;
  }
  return `(데모 번역) ${line}`;
}

const server = http.createServer((req, res) => {
  const send = (obj) => {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(obj));
  };

  if (req.url === '/health') return send({ status: 'healthy', version: 'mock', gpu_available: false });
  if (req.url?.startsWith('/api/sync/')) {
    if (mockSync) return send({ found: true, timestamps: mockSync, lyrics_source: 'mock', language: 'en' });
    return send({ found: false });
  }

  if (req.url === '/api/translate' && req.method === 'POST') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      const { text = '' } = JSON.parse(body || '{}');
      const lines = text.split('\n').map(original => ({
        original,
        translation: translateLine(original),
        pronunciation: null,
      }));
      send({ lines, source_lang: 'en', target_lang: 'ko', engine: 'mock' });
    });
    return;
  }

  res.writeHead(404);
  res.end();
});

server.listen(8000, () => console.log('mock server on :8000'));
