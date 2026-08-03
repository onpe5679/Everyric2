/**
 * PiP 창 안의 가사 패널을 읽는 **한 곳** — 하네스들이 공유한다.
 *
 * 2026-08-04 재작업으로 PiP 창 안의 가사 UI가 메인 가사창과 **같은 인스턴스**
 * (LyricsOverlay, chrome:'filled')가 됐다. 그 전까지 pip.ts가 반쪽으로 다시 구현하던
 * DOM(.ey-pip-stage / .ey-pip-line / .ey-pip-pron / .ey-pip-pitch / .ey-pip-panel)은
 * 전부 사라졌고, 이제 PiP 문서 안에도 `#everyric-root`의 Shadow DOM이 하나 있다 —
 * 셀렉터가 메인 창과 완전히 같아진다는 것 자체가 이 재작업의 목적이다.
 *
 * 하네스마다 이 구조를 따로 적어 두면 다음 개편에서 또 네 곳이 조용히 낡는다.
 * 페이지 컨텍스트에서 평가할 표현식 문자열을 여기서 한 번만 만든다.
 */

/**
 * PiP 패널 상태를 읽는 표현식 — `page.evaluate(readPipPanel())` 형태로 쓴다.
 * PiP가 닫혀 있으면 `{ open: false }`, 열렸지만 패널이 없으면 `{ open: true, panel: false }`.
 */
export function readPipPanel() {
  return `(() => {
    const w = window.documentPictureInPicture?.window;
    if (!w) return { open: false };
    const root = w.document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return { open: true, panel: false };
    // 레인은 **Shadow DOM 밖**에 있다: PiP 창은 [레인][영상][가사창][재생목록] 가로
    // 3열이고, 레인 열은 문서의 light DOM이라 부착 패널이 그리로 나간다(OverlaySlots).
    // floating(메인 창)에서는 여전히 Shadow DOM 안이므로 두 곳을 다 본다.
    const canvas = w.document.querySelector('.ey-main-lane') ?? root.querySelector('.ey-main-lane');
    const laneWrap = canvas?.closest('.ey-lane-wrap') ?? null;
    let lane = { present: false, visible: false, drawnPx: 0, width: 0, height: 0 };
    if (canvas) {
      lane.present = true;
      lane.visible = laneWrap ? w.getComputedStyle(laneWrap).display !== 'none' : false;
      lane.width = canvas.width;
      lane.height = canvas.height;
      try {
        const d = canvas.getContext('2d').getImageData(0, 0, canvas.width || 1, canvas.height || 1).data;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 0) lane.drawnPx++;
      } catch { /* 크기 0 등 — drawnPx 0 유지 */ }
    }
    const active = root.querySelector('.ey-line.active');
    return {
      open: true, panel: true,
      // 예전 .ey-pip-line.current에 해당 — 지금 부르고 있는 줄
      currentLine: active?.querySelector('.ey-line-text')?.textContent?.slice(0, 60)
        ?? active?.textContent?.slice(0, 60) ?? null,
      pron: active?.querySelector('.ey-line-pron')?.textContent?.slice(0, 60) ?? '',
      translation: active?.querySelector('.ey-line-tr')?.textContent?.slice(0, 60) ?? '',
      lineCount: root.querySelectorAll('.ey-line').length,
      filled: !!root.querySelector('.ey-panel.ey-panel-filled'),
      lane,
      // 창 자체의 조각(패널 밖) — 여기는 그대로 남아 있다
      videoMirror: (() => {
        const v = w.document.querySelector('.ey-pip-video');
        return v ? w.getComputedStyle(v).display !== 'none' : false;
      })(),
      hasVolume: !!w.document.querySelector('.ey-pip-volume'),
      hasDivider: !!w.document.querySelector('.ey-pip-divider'),
      volumeValue: w.document.querySelector('.ey-pip-volume')?.value ?? null,
      // 반쪽 구현이 되살아나면 즉시 잡는다(이중 표시 회귀 감시).
      // .ey-pip-stage는 목록에서 뺐다 — 「영상 아래 한 줄」은 운영자가 유지를 지시한
      // 화면이고, 지금은 overlay.attachShortView가 **공용 buildLineEl**로 만든다.
      // 진짜 불변식은 아래 shortUsesSharedLine이다: 그 안이 .ey-line이어야 한다.
      legacyDom: !!w.document.querySelector(
        '.ey-pip-pitch, .ey-pip-panel, .ey-pip-lyricscol, .ey-pip-tr, .ey-pip-pron'),
      /** 단축 표시가 공용 줄 렌더러를 쓰는가 — 세 번째 가사 구현 방지의 핵심 지표 */
      shortUsesSharedLine: !!w.document.querySelector('.ey-pip-stage .ey-pip-line.current .ey-line'),
      /** 3열 구조가 실제로 섰는가 */
      cols: {
        lane: !!w.document.querySelector('.ey-pip-lane-col'),
        center: !!w.document.querySelector('.ey-pip-center'),
        playlist: !!w.document.querySelector('.ey-pip-playlist-col'),
      },
    };
  })()`;
}
