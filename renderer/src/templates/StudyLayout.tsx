import React from "react";
import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { LyricColumn } from "../components/LyricColumn";
import { SpectrumVisualizer } from "../components/SpectrumVisualizer";
import type { CompositionProps, StylePreset } from "../types";
import { findActiveComment } from "../utils/timing";

export const StudyLayout: React.FC<CompositionProps> = ({
  timeline,
  spectrum,
  preset = "minimal",
  showPronunciation = true,
  showComment = true,
  showWordHighlight = true,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  const { meta, lines } = timeline;
  const accent = meta.accent_color || "#d88392";
  const activeNote = findActiveComment(lines, currentTime);

  // Smooth fade-in whenever the active learning comment changes
  const noteTriggerFrame = activeNote ? Math.round(activeNote.lineStart * fps) : 0;
  const noteOpacity = activeNote
    ? spring({
        frame: Math.max(0, frame - noteTriggerFrame),
        fps,
        config: { damping: 20, stiffness: 140 },
      })
    : 0;

  // Gentle ambient scale drift on pre-blurred background
  const bgScale = interpolate(Math.sin(frame * 0.015), [-1, 1], [1.02, 1.07]);

  const isEditorial = preset === "editorial";
  const isClassic = preset === "classic2";
  const isMinimal = preset === "minimal";

  const coverSize = isEditorial ? 128 : isClassic ? 260 : 320;
  const contentScale = Math.max(0.85, Math.min(1.4, meta.content_scale ?? 1.18));

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0c0a0f",
        color: "#f3efe9",
        fontFamily:
          "'Noto Sans KR', 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif",
      }}
    >
      {/* Pre-blurred 1920x1080 background image + soft ambient color wash */}
      <AbsoluteFill style={{ overflow: "hidden" }}>
        <Img
          src={staticFile("track/bg_blur.png")}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: `scale(${bgScale})`,
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: `radial-gradient(circle at 28% 45%, ${accent}33, transparent 54%), radial-gradient(circle at 72% 58%, rgba(56, 72, 115, 0.18), transparent 56%)`,
          }}
        />
      </AbsoluteFill>

      {/* Safe-margin centered grid container matching HTML preview proportions */}
      <AbsoluteFill
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: `${(100 / contentScale).toFixed(3)}%`,
            height: `${(100 / contentScale).toFixed(3)}%`,
            transform: `scale(${contentScale})`,
            transformOrigin: "center center",
            padding: isClassic ? "64px 100px" : "70px 115px",
            boxSizing: "border-box",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: "100%",
              height: "100%",
              display: "grid",
              gridTemplateColumns: "34% 66%",
              gap: isClassic ? 68 : 84,
              alignItems: "center",
            }}
          >
          {/* Left Column: Artwork + Metadata + Spectrum/Wave + Learning Note */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: isEditorial ? "flex-start" : "center",
              textAlign: isEditorial ? "left" : "center",
              justifyContent: "center",
              gap: isEditorial ? 24 : 0,
            }}
          >
            {isEditorial ? (
              <div style={{ display: "flex", alignItems: "center", gap: 22 }}>
                <Img
                  src={staticFile("track/cover.jpg")}
                  style={{
                    width: coverSize,
                    height: coverSize,
                    borderRadius: 10,
                    objectFit: "cover",
                    border: "1px solid rgba(255,255,255,0.14)",
                    boxShadow: "0 18px 40px rgba(0,0,0,0.65)",
                  }}
                />
                <div>
                  <div
                    style={{
                      fontFamily: "'Noto Serif KR', Georgia, serif",
                      fontSize: 28,
                      fontWeight: 700,
                      color: "#f5f1eb",
                    }}
                  >
                    {meta.title}
                  </div>
                  <div style={{ fontSize: 18, color: "#9f9690", marginTop: 4 }}>
                    {meta.artist}
                  </div>
                </div>
              </div>
            ) : (
              <>
                <Img
                  src={staticFile("track/cover.jpg")}
                  style={{
                    width: coverSize,
                    height: coverSize,
                    borderRadius: isClassic ? 4 : 14,
                    objectFit: "cover",
                    border: isClassic
                      ? `1px solid ${accent}44`
                      : "1px solid rgba(255,255,255,0.12)",
                    boxShadow: "0 24px 60px rgba(0,0,0,0.7)",
                  }}
                />
                <div
                  style={{
                    fontFamily: "'Noto Serif KR', Georgia, serif",
                    fontSize: isClassic ? 32 : 30,
                    fontWeight: 700,
                    letterSpacing: isClassic ? "0.1em" : "0.02em",
                    color: "#f5f1eb",
                    marginTop: 24,
                  }}
                >
                  {meta.title}
                </div>
                {isMinimal && (
                  <div style={{ fontSize: 19, color: "#a59c96", marginTop: 6 }}>
                    {meta.artist}
                  </div>
                )}
              </>
            )}

            {/* Audio Spectrum / Mini Wave */}
            <div
              style={{
                marginTop: isEditorial ? 0 : 22,
                marginBottom: isMinimal ? 22 : 0,
              }}
            >
              <SpectrumVisualizer
                spectrum={spectrum}
                mode={isMinimal ? "mini-wave" : "led-bars"}
                accentColor={accent}
                width={isMinimal ? 300 : isEditorial ? 400 : 320}
                height={isMinimal ? 38 : 48}
              />
            </div>

            {isClassic && (
              <div
                style={{
                  fontSize: 19,
                  color: "#9c938e",
                  marginTop: 12,
                  letterSpacing: "0.03em",
                }}
              >
                {meta.artist}
              </div>
            )}

            {/* Language Learning Commentary Slot */}
            {showComment && activeNote ? (
              <div
                style={{
                  width: "100%",
                  maxWidth: isEditorial ? 460 : 440,
                  marginTop: isEditorial ? 8 : 24,
                  paddingTop: isEditorial ? 2 : 20,
                  paddingLeft: isEditorial ? 22 : 0,
                  borderTop: isEditorial
                    ? "none"
                    : "1px solid rgba(255,255,255,0.16)",
                  borderLeft: isEditorial ? `3px solid ${accent}` : "none",
                  textAlign: isClassic ? "center" : "left",
                  opacity: noteOpacity,
                  transform: `translateY(${(1 - noteOpacity) * 8}px)`,
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 700,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: accent,
                  }}
                >
                  {activeNote.comment.tag}
                </div>
                <div
                  style={{
                    fontSize: 25,
                    fontWeight: 700,
                    color: "#f5f1eb",
                    marginTop: 6,
                  }}
                >
                  {activeNote.comment.phrase}
                </div>
                <div
                  style={{
                    fontSize: 19,
                    color: "#dcd4ce",
                    marginTop: 5,
                    lineHeight: 1.4,
                    wordBreak: "keep-all",
                  }}
                >
                  {activeNote.comment.meaning}
                </div>
                {!isClassic && activeNote.comment.tip ? (
                  <div
                    style={{
                      fontSize: 16,
                      color: "#9c938e",
                      marginTop: 9,
                      lineHeight: 1.45,
                      wordBreak: "keep-all",
                    }}
                  >
                    {activeNote.comment.tip}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          {/* Right Column: Masked Scrolling Lyrics */}
          <LyricColumn
            lines={lines}
            stylePreset={preset}
            accentColor={accent}
            showPronunciation={showPronunciation}
            showWordHighlight={showWordHighlight}
          />
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
