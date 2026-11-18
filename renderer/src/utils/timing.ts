import type { LearningComment, LyricLine } from "../types";

export const SCROLL_DURATION_SEC = 0.28;

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

/**
 * Determine which line index the lyric scroller should center on.
 * Advances SCROLL_DURATION_SEC ahead of line.start so the scroll completes right at line.start.
 */
export function findScrollTargetIndex(lines: LyricLine[], currentTimeSec: number): number {
  if (lines.length === 0) return 0;
  const effectiveTime = currentTimeSec + SCROLL_DURATION_SEC;
  if (effectiveTime < lines[0].start) return 0;

  for (let i = 0; i < lines.length; i++) {
    if (effectiveTime >= lines[i].start && effectiveTime < lines[i].end) {
      return i;
    }
    if (i + 1 < lines.length && effectiveTime >= lines[i].end && effectiveTime < lines[i + 1].start) {
      return i;
    }
  }
  return lines.length - 1;
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

/**
 * Smoothly interpolate scroll position using Remotion's spring().
 */
export function computeSmoothLineOffset(
  lines: LyricLine[],
  targetIdx: number,
  frame: number,
  fps: number,
): number {
  if (targetIdx <= 0 || lines.length === 0) return 0;
  const prevIdx = targetIdx - 1;
  const currentTimeSec = frame / Math.max(1, fps);
  const arriveTime = lines[targetIdx].start;
  const startTime = Math.max(0, arriveTime - SCROLL_DURATION_SEC);
  const span = Math.max(0.001, arriveTime - startTime);
  const u = Math.max(0, Math.min(1, (currentTimeSec - startTime) / span));
  const progress = 1 - Math.pow(1 - u, 3);
  return prevIdx + progress;
}
