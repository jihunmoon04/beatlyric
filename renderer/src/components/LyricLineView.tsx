import React from "react";
import type { LyricLine, StylePreset } from "../types";

interface Props {
  line: LyricLine;
  isActive: boolean;
  currentTime: number;
  stylePreset: StylePreset;
  accentColor: string;
  showPronunciation: boolean;
  showWordHighlight: boolean;
}

export const LyricLineView: React.FC<Props> = ({
  line,
  isActive,
  currentTime,
  stylePreset,
  accentColor,
  showPronunciation,
  showWordHighlight,
}) => {
  const renderOriginalText = () => {
    if (!isActive || line.whole_line || !showWordHighlight || !line.words?.length) {
      return line.original;
    }
    return line.words.map((w, idx) => {
      const spoken = currentTime >= w.start;
      const current = currentTime >= w.start && currentTime <= w.end;
      const color = current
        ? "#f7dce0"
        : spoken
          ? "#f5f1eb"
          : "rgba(245, 241, 235, 0.42)";
      return (
        <span key={idx} style={{ color }}>
          {w.text}
          {idx < line.words.length - 1 ? " " : ""}
        </span>
      );
    });
  };

  if (stylePreset === "minimal") {
    return (
      <div
        style={{
          textAlign: "left",
          opacity: isActive ? 1 : 0.36,
          filter: isActive ? "none" : "blur(0.7px)",
          transform: isActive ? "scale(1.015)" : "scale(1)",
          transformOrigin: "left center",
          padding: "6px 12px",
        }}
      >
        <div
          style={{
            fontSize: 34,
            fontWeight: 700,
            lineHeight: 1.24,
            letterSpacing: "-0.01em",
            whiteSpace: "normal",
            overflowWrap: "break-word",
            wordBreak: "keep-all",
            color: "#f5f1eb",
          }}
        >
          {renderOriginalText()}
        </div>
        {showPronunciation && line.pronunciation ? (
          <div
            style={{
              fontSize: 19,
              marginTop: 6,
              whiteSpace: "normal",
              overflowWrap: "break-word",
              wordBreak: "keep-all",
              color: isActive ? accentColor : "#a79d97",
            }}
          >
            {line.pronunciation}
          </div>
        ) : null}
        <div
          style={{
            fontSize: 24,
            marginTop: 5,
            whiteSpace: "normal",
            overflowWrap: "break-word",
            wordBreak: "keep-all",
            color: isActive ? "#e2dbd5" : "#8f8680",
          }}
        >
          {line.translation}
        </div>
      </div>
    );
  }

  // Editorial & Classic2 share refined serif primary + border-left active highlight
  const isCentered = stylePreset === "classic2";

  return (
    <div
      style={{
        textAlign: isCentered ? "center" : "left",
        opacity: isActive ? 1 : 0.34,
        backgroundColor: isActive ? "rgba(56, 19, 31, 0.62)" : "transparent",
        borderLeft: isActive ? `4px solid ${accentColor}` : "4px solid transparent",
        padding: "16px 26px",
        borderRadius: isCentered ? 2 : 6,
      }}
    >
      <div
        style={{
          fontFamily: "'Noto Serif KR', Georgia, 'Times New Roman', serif",
          fontSize: isCentered ? 34 : 32,
          fontWeight: 700,
          lineHeight: 1.28,
          whiteSpace: "normal",
          overflowWrap: "break-word",
          wordBreak: "keep-all",
          color: "#f5f1eb",
        }}
      >
        {renderOriginalText()}
      </div>
      {showPronunciation && line.pronunciation ? (
        <div
          style={{
            fontSize: 19,
            marginTop: 7,
            whiteSpace: "normal",
            overflowWrap: "break-word",
            wordBreak: "keep-all",
            color: isActive ? accentColor : "#9c928d",
          }}
        >
          {line.pronunciation}
        </div>
      ) : null}
      <div
        style={{
          fontSize: 23,
          marginTop: 5,
          whiteSpace: "normal",
          overflowWrap: "break-word",
          wordBreak: "keep-all",
          color: isActive ? "#e6dfd9" : "#867d78",
        }}
      >
        {line.translation}
      </div>
    </div>
  );
};
