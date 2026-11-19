import type { LearningComment, LyricLine, SpectrumData, StylePreset } from "../types";

export const SCROLL_DURATION_SEC = 0.42;
export const LINE_FADE_SEC = 0.18;

export interface LineLayoutMetrics {
  heights: number[];
  tops: number[];
  centers: number[];
  totalHeight: number;
}

/**
 * Format seconds into M:SS clock string.
 */
export function formatClock(sec: number): string {
  const safe = Math.max(0, Math.floor(sec));
  const mins = Math.floor(safe / 60);
  const rem = safe % 60;
  return `${mins}:${String(rem).padStart(2, "0")}`;
}

/**
 * Estimate visual width units accounting for wider CJK glyphs vs Latin characters.
 */
export function estimateVisualUnits(text: string): number {
  let units = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0x1100) {
      units += 1.75;
    } else if (code >= 65 && code <= 90) {
      units += 1.08;
    } else {
      units += 0.95;
    }
  }
  return units;
}

/**
 * Compute variable row heights, top offsets, and center coordinates calibrated per preset.
 */
export function computeLineLayoutMetrics(
  lines: LyricLine[],
  opts: {
    stylePreset: StylePreset;
    showPronunciation: boolean;
    contentScale: number;
  },
): LineLayoutMetrics {
  const { stylePreset, showPronunciation, contentScale } = opts;
  const scale = Math.max(0.85, Math.min(1.4, contentScale || 1.18));

  const cfg =
    stylePreset === "editorial"
      ? {
          colWidthPx: Math.max(400, (1920 / scale - 192) * 0.63 - 64 - 56),
          origPx: 32,
          origLh: 1.3,
          pronPx: 19,
          pronLh: 1.35,
          transPx: 23,
          transLh: 1.34,
          vertPad: 58,
          minHeight: 144,
        }
      : stylePreset === "classic2"
        ? {
            colWidthPx: Math.max(400, (1920 / scale - 192) * 0.62 - 64 - 56),
            origPx: 34,
            origLh: 1.3,
            pronPx: 19,
            pronLh: 1.35,
            transPx: 23,
            transLh: 1.34,
            vertPad: 62,
            minHeight: 148,
          }
        : stylePreset === "shorts"
          ? {
              colWidthPx: Math.max(480, 1080 / scale - 68 - 96 - 24),
              origPx: 42,
              origLh: 1.26,
              pronPx: 23,
              pronLh: 1.35,
              transPx: 28,
              transLh: 1.34,
              vertPad: 56,
              minHeight: 164,
            }
          : {
              colWidthPx: Math.max(420, (1920 / scale - 216) * 0.65 - 80 - 24),
              origPx: 34,
              origLh: 1.26,
              pronPx: 19,
              pronLh: 1.35,
              transPx: 24,
              transLh: 1.34,
              vertPad: 44,
              minHeight: 132,
            };

  const wrappedLines = (text: string | undefined, fontPx: number): number => {
    const clean = (text || "").trim();
    if (!clean) return 0;
    const charsPerLine = Math.max(10, cfg.colWidthPx / (fontPx * 0.56));
    return Math.max(1, Math.ceil((estimateVisualUnits(clean) * 1.08) / charsPerLine));
  };

  const heights: number[] = [];
  const tops: number[] = [];
  const centers: number[] = [];
  let cursor = 0;

  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const origRows = wrappedLines(ln.original, cfg.origPx);
    const pronRows = showPronunciation ? wrappedLines(ln.pronunciation, cfg.pronPx) : 0;
    const transRows = wrappedLines(ln.translation, cfg.transPx);

    const rawHeight =
      origRows * cfg.origPx * cfg.origLh +
      (pronRows > 0 ? 7 + pronRows * cfg.pronPx * cfg.pronLh : 0) +
      (transRows > 0 ? 6 + transRows * cfg.transPx * cfg.transLh : 0) +
      cfg.vertPad;

    const h = Math.max(cfg.minHeight, Math.ceil(rawHeight));
    heights.push(h);
    tops.push(cursor);
    centers.push(cursor + h / 2);
    cursor += h;
  }

  return {
    heights,
    tops,
    centers,
    totalHeight: cursor,
  };
}

function scrollStart(lines: LyricLine[], index: number, lead = SCROLL_DURATION_SEC): number {
  return Math.max(0, lines[index - 1].start, lines[index].start - lead);
}

function scrollProgress(lines: LyricLine[], index: number, currentTimeSec: number): number {
  const start = scrollStart(lines, index);
  const end = lines[index].start;
  if (currentTimeSec >= end) return 1;
  const u = Math.max(0, (currentTimeSec - start) / Math.max(0.001, end - start));
  return u * u * (3 - 2 * u);
}

/**
 * Smoothly interpolate pixel Y scroll center across variable-height lyric rows.
 */
export function computeSmoothScrollY(
  lines: LyricLine[],
  centers: number[],
  targetIdx: number,
  currentTimeSec: number,
): number {
  if (lines.length === 0 || centers.length === 0) return 0;
  if (targetIdx <= 0) return centers[0] ?? 0;
  const safeIdx = Math.min(targetIdx, centers.length - 1);
  const prevCenter = centers[safeIdx - 1] ?? 0;
  const targetCenter = centers[safeIdx] ?? prevCenter;
  return prevCenter + (targetCenter - prevCenter) * scrollProgress(lines, safeIdx, currentTimeSec);
}

/**
 * Average low-frequency bass bars (0..3) at the given frame, clamped to [0, 1].
 */
export function computeBassEnergy(spectrum: SpectrumData, frame: number): number {
  if (!spectrum.frames || spectrum.frames.length === 0) return 0;
  const safeIdx = Math.min(Math.max(0, frame), spectrum.frames.length - 1);
  const bars = spectrum.frames[safeIdx];
  if (!bars || bars.length === 0) return 0;
  const count = Math.min(4, bars.length);
  let sum = 0;
  for (let i = 0; i < count; i++) {
    sum += Number(bars[i] || 0);
  }
  return Math.max(0, Math.min(1, sum / count));
}

/**
 * Compute smooth ease-out cubic activation ∈ [0, 1] around line.start and line.end.
 */
export function computeLineActivation(
  line: LyricLine,
  currentTimeSec: number,
  fadeSec: number = LINE_FADE_SEC,
): number {
  const halfFade = fadeSec * 0.5;
  const inStart = line.start - halfFade;
  const inEnd = Math.min(line.end, line.start + halfFade);
  const outStart = line.end;
  const outEnd = line.end + fadeSec;

  if (currentTimeSec <= inStart || currentTimeSec >= outEnd) {
    return 0;
  }
  if (currentTimeSec < inEnd) {
    const u = Math.max(0, Math.min(1, (currentTimeSec - inStart) / Math.max(0.001, inEnd - inStart)));
    return 1 - Math.pow(1 - u, 3);
  }
  if (currentTimeSec <= outStart) {
    return 1;
  }
  const u = Math.max(0, Math.min(1, (currentTimeSec - outStart) / Math.max(0.001, outEnd - outStart)));
  return 1 - (1 - Math.pow(1 - u, 3));
}

/**
 * Return the index of the line currently being sung, or -1 during intro/instrumental breaks.
 */
export function findActiveLineIndex(lines: LyricLine[], currentTimeSec: number): number {
  for (let i = 0; i < lines.length; i++) {
    if (currentTimeSec >= lines[i].start && currentTimeSec < lines[i].end) {
      return i;
    }
  }
  return -1;
}

/** Find the last lyric row whose scroll transition has started. */
export function findScrollTargetIndex(lines: LyricLine[], currentTimeSec: number, lead = SCROLL_DURATION_SEC): number {
  let lower = 1;
  let upper = lines.length;
  while (lower < upper) {
    const mid = Math.floor((lower + upper) / 2);
    if (currentTimeSec >= scrollStart(lines, mid, lead)) lower = mid + 1;
    else upper = mid;
  }
  return lower - 1;
}

export function findActiveComment(
  lines: LyricLine[],
  currentTimeSec: number,
): { comment: LearningComment; lineStart: number; isInstrumental: boolean } | null {
  if (lines.length === 0) return null;
  const activeIdx = findActiveLineIndex(lines, currentTimeSec);

  // If currently singing, walk back to the most recent non-null comment up to activeIdx
  if (activeIdx >= 0) {
    for (let i = activeIdx; i >= 0; i--) {
      const c = lines[i].comment;
      if (c) {
        return { comment: c, lineStart: lines[i].start, isInstrumental: false };
      }
    }
    return null;
  }

  // In a gap/instrumental: find the latest finished line
  let lastFinished = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].end <= currentTimeSec) {
      lastFinished = i;
    }
  }
  if (lastFinished < 0) {
    // Intro before first lyric
    const firstComment = lines[0].comment;
    return firstComment ? { comment: firstComment, lineStart: lines[0].start, isInstrumental: true } : null;
  }

  for (let i = lastFinished; i >= 0; i--) {
    const c = lines[i].comment;
    if (c) {
      return { comment: c, lineStart: lines[i].start, isInstrumental: true };
    }
  }
  return null;
}

/** Smoothly interpolate scroll position in rows. */
export function computeSmoothLineOffset(
  lines: LyricLine[],
  targetIdx: number,
  frame: number,
  fps: number,
): number {
  if (targetIdx <= 0 || lines.length === 0) return 0;
  return targetIdx - 1 + scrollProgress(lines, targetIdx, frame / Math.max(1, fps));
}
