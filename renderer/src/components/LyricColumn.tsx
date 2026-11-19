import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import type { LyricLine, StylePreset } from "../types";
import {
  computeLineLayoutMetrics,
  computeSmoothScrollY,
  findActiveLineIndex,
  findScrollTargetIndex,
} from "../utils/timing";
import { LyricLineView } from "./LyricLineView";

interface Props {
  lines: LyricLine[];
  stylePreset: StylePreset;
  accentColor: string;
  showPronunciation: boolean;
  showWordHighlight: boolean;
  karaokeSweep: boolean;
  smoothLyricScroll: boolean;
  lyricWordLift: boolean;
  contentScale: number;
}

export const LyricColumn: React.FC<Props> = ({
  lines,
  stylePreset,
  accentColor,
  showPronunciation,
  showWordHighlight,
  karaokeSweep,
  smoothLyricScroll,
  lyricWordLift,
  contentScale,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  const activeIdx = findActiveLineIndex(lines, currentTime);
  const targetIdx = findScrollTargetIndex(lines, currentTime, smoothLyricScroll ? undefined : 0);
  const metrics = computeLineLayoutMetrics(lines, {
    stylePreset,
    showPronunciation,
    contentScale,
  });
  const scrollY = smoothLyricScroll
    ? computeSmoothScrollY(lines, metrics.centers, targetIdx, currentTime)
    : (metrics.centers[targetIdx] ?? 0);

  return (
    <div
      style={{
        position: "relative",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          overflow: "hidden",
          WebkitMaskImage:
            "linear-gradient(to bottom, transparent 0%, black 12%, black 88%, transparent 100%)",
          maskImage:
            "linear-gradient(to bottom, transparent 0%, black 12%, black 88%, transparent 100%)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "50%",
            transform: `translateY(${(-scrollY).toFixed(2)}px)`,
          }}
        >
          {lines.map((line, idx) => (
            <div
              key={line.id}
              style={{
                height: metrics.heights[idx],
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              <LyricLineView
                line={line}
                isActive={idx === activeIdx}
                currentTime={currentTime}
                stylePreset={stylePreset}
                accentColor={accentColor}
                showPronunciation={showPronunciation}
                showWordHighlight={showWordHighlight}
                karaokeSweep={karaokeSweep}
                lyricWordLift={lyricWordLift}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
