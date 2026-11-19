import React from "react";
import { useCurrentFrame } from "remotion";
import type { SpectrumData } from "../types";

interface Props {
  spectrum: SpectrumData;
  mode: "mini-wave" | "led-bars";
  accentColor?: string;
  width: number;
  height: number;
}

export const SpectrumVisualizer: React.FC<Props> = ({
  spectrum,
  mode,
  accentColor = "#d88392",
  width,
  height,
}) => {
  const frame = useCurrentFrame();
  const safeIdx = Math.min(
    Math.max(0, frame),
    Math.max(0, spectrum.frames.length - 1),
  );
  const bars = spectrum.frames[safeIdx] ?? new Array(28).fill(0.15);
  const count = bars.length;

  if (mode === "mini-wave") {
    const gap = 5;
    const barW = (width - gap * (count - 1)) / count;
    const centerY = height / 2;

    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
        {bars.map((_, i) => {
          const distFromCenter = Math.abs(i - (count - 1) / 2) / (count / 2);
          const binIdx = Math.min(count - 1, Math.floor(distFromCenter * (count - 1)));
          const val = bars[binIdx] ?? 0.15;
          const envelope = 1 - Math.pow(distFromCenter, 1.75);
          const barH = Math.max(3, (0.14 + val * 0.86) * envelope * (height - 2));
          const x = i * (barW + gap);
          const y = centerY - barH / 2;
          const opacity = 0.35 + 0.55 * envelope;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={barH}
              rx={barW / 2}
              fill={accentColor}
              fillOpacity={opacity}
            />
          );
        })}
      </svg>
    );
  }

  // Segmented LED bar spectrum (Editorial & Classic2)
  const gap = 5;
  const barW = (width - gap * (count - 1)) / count;
  const segH = 4;
  const segGap = 3;
  const maxSegments = Math.floor(height / (segH + segGap));

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {bars.map((val, i) => {
        const activeSegs = Math.max(1, Math.round((0.08 + val * 0.92) * maxSegments));
        const x = i * (barW + gap);
        return Array.from({ length: activeSegs }).map((_, s) => {
          const y = height - (s + 1) * (segH + segGap);
          const ratio = s / Math.max(1, maxSegments);
          const fillOpacity = ratio > 0.65 ? 0.95 : ratio > 0.3 ? 0.78 : 0.58;
          return (
            <rect
              key={`${i}-${s}`}
              x={x}
              y={y}
              width={barW}
              height={segH}
              rx={1}
              fill={accentColor}
              fillOpacity={fillOpacity}
            />
          );
        });
      })}
    </svg>
  );
};
