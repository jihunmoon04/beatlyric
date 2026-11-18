import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import type { LyricLine, StylePreset } from "../types";
import {
  computeSmoothLineOffset,
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
}

const ITEM_HEIGHT_BY_STYLE: Record<StylePreset, number> = {
  minimal: 210,
  editorial: 210,
  classic2: 210,
};

export const LyricColumn: React.FC<Props> = ({
  lines,
  stylePreset,
  accentColor,
  showPronunciation,
  showWordHighlight,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  const activeIdx = findActiveLineIndex(lines, currentTime);
  const targetIdx = findScrollTargetIndex(lines, currentTime);
  const smoothIdx = computeSmoothLineOffset(lines, targetIdx, frame, fps);
  const itemHeight = ITEM_HEIGHT_BY_STYLE[stylePreset];
  const translateY = -(smoothIdx * itemHeight + itemHeight / 2);

  return (
    <div
      style={{
        position: "relative",
        height: "100%",
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
          transform: `translateY(${translateY}px)`,
        }}
      >
        {lines.map((line, idx) => (
          <div
            key={line.id}
            style={{
              height: itemHeight,
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
            />
          </div>
        ))}
      </div>
    </div>
  );
};
