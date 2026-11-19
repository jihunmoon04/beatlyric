import React from "react";
import type { LyricLine, StylePreset } from "../types";
import { computeLineActivation } from "../utils/timing";

interface Props {
  line: LyricLine;
  isActive: boolean;
  currentTime: number;
  stylePreset: StylePreset;
  accentColor: string;
  showPronunciation: boolean;
  showWordHighlight: boolean;
  karaokeSweep: boolean;
  lyricWordLift: boolean;
}

export const LyricLineView: React.FC<Props> = ({
  line,
  isActive,
  currentTime,
  stylePreset,
  accentColor,
  showPronunciation,
  showWordHighlight,
  karaokeSweep,
  lyricWordLift,
}) => {
  const activation = computeLineActivation(line, currentTime);

  const renderOriginalText = () => {
    if (line.whole_line || !showWordHighlight || !line.words?.length) return line.original;

    const fillTint = `color-mix(in oklab, ${accentColor} 38%, #ffffff)`;
    const activeTint = `color-mix(in oklab, ${accentColor} 42%, #ffffff)`;
    return line.words.map((word, index) => {
      const highlightStart = index === 0 ? line.start : word.start;
      const canLift = karaokeSweep && lyricWordLift && word.text.length <= 16;
      const style: React.CSSProperties = { display: canLift ? "inline-block" : undefined };

      if (isActive) {
        if (currentTime < highlightStart) {
          style.color = karaokeSweep ? "rgba(245, 241, 235, 0.40)" : "rgba(245, 241, 235, 0.42)";
        } else if (currentTime >= word.end) {
          style.color = "#f5f1eb";
        } else if (karaokeSweep) {
          const progress = Math.min(1, (currentTime - highlightStart) / Math.max(0.04, word.end - highlightStart));
          const stopA = Math.max(0, progress * 100 - 4).toFixed(1);
          const stopB = Math.min(100, progress * 100 + 6).toFixed(1);
          const lift = canLift ? Math.sin(progress * Math.PI) * 2.2 : 0;
          style.backgroundImage = `linear-gradient(90deg, ${fillTint} 0%, ${fillTint} ${stopA}%, rgba(245, 241, 235, 0.40) ${stopB}%, rgba(245, 241, 235, 0.40) 100%)`;
          style.WebkitBackgroundClip = "text";
          style.backgroundClip = "text";
          style.color = "transparent";
          style.WebkitTextFillColor = "transparent";
          if (lift > 0.05) style.transform = `translateY(-${lift.toFixed(2)}px)`;
          style.filter = `drop-shadow(0 0 10px color-mix(in oklab, ${accentColor} 55%, transparent))`;
        } else {
          style.color = activeTint;
        }
      }

      return (
        <React.Fragment key={index}>
          <span style={style}>{word.text}</span>
          {index < line.words.length - 1 ? " " : ""}
        </React.Fragment>
      );
    });
  };

  if (stylePreset === "minimal" || stylePreset === "shorts") {
    const isShorts = stylePreset === "shorts";
    const blurPx = (1 - activation) * (isShorts ? 0.6 : 0.7);
    const baseOpacity = isShorts ? 0.32 + 0.68 * activation : 0.36 + 0.64 * activation;
    const scaleVal = isShorts ? 1 + 0.02 * activation : 1 + 0.015 * activation;

    return (
      <div
        style={{
          textAlign: "left",
          opacity: Number(baseOpacity.toFixed(3)),
          filter: blurPx < 0.02 ? "none" : `blur(${blurPx.toFixed(2)}px)`,
          transform: `scale(${scaleVal.toFixed(4)})`,
          transformOrigin: "left center",
          padding: isShorts ? "8px 12px" : "6px 12px",
        }}
      >
        <div
          style={{
            fontSize: isShorts ? 42 : 34,
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
              fontSize: isShorts ? 23 : 19,
              marginTop: isShorts ? 8 : 6,
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
            fontSize: isShorts ? 28 : 24,
            marginTop: isShorts ? 6 : 5,
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

  if (stylePreset === "classic2") {
    const hairline = `1px solid color-mix(in oklab, ${accentColor} ${Math.round(activation * 55)}%, transparent)`;
    const backdrop =
      activation > 0.01
        ? `linear-gradient(90deg var(--in-oklab), transparent, color-mix(in oklab, ${accentColor} ${Math.round(activation * 14)}%, rgba(16, 13, 20, ${(activation * 0.55).toFixed(3)})), transparent)`
        : "transparent";

    return (
      <div
        style={{
          textAlign: "center",
          opacity: Number((0.34 + 0.66 * activation).toFixed(3)),
          background: backdrop,
          borderTop: hairline,
          borderBottom: hairline,
          padding: "16px 28px",
        }}
      >
        <div
          className="study-serif"
          style={{
            fontSize: 34,
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
  }

  // Editorial: left-aligned serif + accent-tinted frosted card + 4px left border
  const bgStyle =
    activation > 0.01
      ? `color-mix(in oklab, ${accentColor} ${Math.round(activation * 22)}%, rgba(16, 13, 20, ${(activation * 0.62).toFixed(3)}))`
      : "transparent";

  return (
    <div
      style={{
        textAlign: "left",
        opacity: Number((0.34 + 0.66 * activation).toFixed(3)),
        backgroundColor: bgStyle,
        borderLeft: `4px solid color-mix(in oklab, ${accentColor} ${Math.round(activation * 100)}%, transparent)`,
        padding: "16px 26px",
        borderRadius: 6,
      }}
    >
      <div
        className="study-serif"
        style={{
          fontSize: 32,
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
