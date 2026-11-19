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
import type { CompositionProps, LearningComment } from "../types";
import {
  computeBassEnergy,
  findActiveComment,
  formatClock,
} from "../utils/timing";

interface ProgressBarProps {
  elapsed: number;
  duration: number;
  accent: string;
  width?: number | string;
}

const TrackProgressBar: React.FC<ProgressBarProps> = ({
  elapsed,
  duration,
  accent,
  width = "100%",
}) => {
  return (
    <div style={{ width, display: "flex", flexDirection: "column", gap: 6 }}>
      <progress
        className="track-progress"
        value={elapsed}
        max={duration}
        aria-label="Track progress"
        style={{ "--progress-bar-fill": accent } as React.CSSProperties}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          fontWeight: 500,
          fontVariantNumeric: "tabular-nums",
          color: "#9f9690",
        }}
      >
        <span>{formatClock(elapsed)}</span>
        <span>{formatClock(duration)}</span>
      </div>
    </div>
  );
};

interface NoteBodyProps {
  comment: LearningComment | null | undefined;
  accent: string;
  opacity: number;
  centered?: boolean;
  hideTip?: boolean;
}

const NoteBody: React.FC<NoteBodyProps> = ({
  comment,
  accent,
  opacity,
  centered = false,
  hideTip = false,
}) => {
  if (!comment) return null;
  return (
    <div
      style={{
        textAlign: centered ? "center" : "left",
        opacity,
        transform: `translateY(${((1 - opacity) * 8).toFixed(2)}px)`,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 700,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: accent,
        }}
      >
        {comment.tag}
      </div>
      <div
        style={{
          fontSize: 24,
          fontWeight: 700,
          color: "#f5f1eb",
          marginTop: 6,
          lineHeight: 1.25,
          wordBreak: "keep-all",
        }}
      >
        {comment.phrase}
      </div>
      <div
        style={{
          fontSize: 18,
          color: "#dcd4ce",
          marginTop: 5,
          lineHeight: 1.4,
          wordBreak: "keep-all",
        }}
      >
        {comment.meaning}
      </div>
      {!hideTip && comment.tip ? (
        <div
          style={{
            fontSize: 15,
            color: "#9c938e",
            marginTop: 8,
            lineHeight: 1.45,
            wordBreak: "keep-all",
          }}
        >
          {comment.tip}
        </div>
      ) : null}
    </div>
  );
};

export const StudyLayout: React.FC<CompositionProps> = ({
  timeline,
  spectrum,
  preset = "minimal",
  showPronunciation = true,
  showComment = true,
  showWordHighlight = true,
  karaokeSweep = true,
  smoothLyricScroll = true,
  lyricWordLift = true,
  showSongContext = true,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  const { meta, lines } = timeline;
  const accent = meta.accent_color || "#d88392";
  const context = meta.song_context;
  const hasSongNotes = Boolean(context?.meaning || context?.culture || lines.some(line => line.culture_note));
  const showNotes = showComment || (showSongContext && hasSongNotes);
  const activeLine = lines.find(line => line.start <= currentTime && currentTime < line.end);
  const instrumental = !activeLine || activeLine.original.startsWith("♪");
  let currentSongNote: LearningComment | undefined;
  if (showSongContext) {
    if ((currentTime < 3 || instrumental) && (context?.meaning || context?.culture)) {
      currentSongNote = { tag: "곡 해설", phrase: context.meaning ?? "", meaning: context.culture ?? "" };
    } else if (activeLine?.culture_note) {
      currentSongNote = { tag: "구절 해설", phrase: "", meaning: activeLine.culture_note };
    }
  }
  const activeNote = currentSongNote
    ? { comment: currentSongNote, lineStart: activeLine?.start ?? 0 }
    : showComment ? findActiveComment(lines, currentTime) : null;

  const noteTriggerFrame = activeNote ? Math.round(activeNote.lineStart * fps) : 0;
  const noteOpacity = activeNote
    ? spring({
        frame: Math.max(0, frame - noteTriggerFrame),
        fps,
        config: { damping: 20, stiffness: 140 },
      })
    : 0;

  const bgScale = interpolate(Math.sin(frame * 0.015), [-1, 1], [1.02, 1.07]);
  const bassEnergy = computeBassEnergy(spectrum, frame);
  const duration = Math.max(0.1, meta.duration || 1);
  const elapsed = Math.min(duration, Math.max(0, currentTime));

  const isEditorial = preset === "editorial";
  const isClassic = preset === "classic2";
  const isShorts = preset === "shorts";

  const rawScale = meta.content_scale ?? 1.18;
  const effectiveScale = isShorts
    ? Math.max(0.88, Math.min(1.12, rawScale * 0.88))
    : Math.max(0.85, Math.min(1.4, rawScale));

  const haloAlphaHex = Math.round(28 + bassEnergy * 48)
    .toString(16)
    .padStart(2, "0");
  const haloSpread = Math.round(18 + bassEnergy * 36);

  return (
    <AbsoluteFill
      className="study-stage-root"
      style={{
        backgroundColor: "#0c0a0f",
        color: "#f3efe9",
      }}
    >
      {/* Pre-blurred background image + soft ambient color wash */}
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
            background: isShorts
              ? `radial-gradient(circle at 50% 22%, ${accent}38, transparent 52%), radial-gradient(circle at 50% 82%, rgba(56, 72, 115, 0.22), transparent 56%)`
              : `radial-gradient(circle at 28% 45%, ${accent}33, transparent 54%), radial-gradient(circle at 72% 58%, rgba(56, 72, 115, 0.18), transparent 56%)`,
          }}
        />
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: `${(100 / effectiveScale).toFixed(3)}%`,
            height: `${(100 / effectiveScale).toFixed(3)}%`,
            transform: `scale(${effectiveScale})`,
            transformOrigin: "center center",
            padding: isShorts
              ? "104px 96px 210px 68px"
              : isClassic
                ? "60px 96px"
                : isEditorial
                  ? "64px 96px"
                  : "64px 108px",
            boxSizing: "border-box",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {isShorts ? (
            /* 9:16 Vertical Shorts Layout */
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                gap: 28,
              }}
            >
              {/* Compact Mobile Header */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                  <Img
                    src={staticFile("track/cover.jpg")}
                    style={{
                      width: 112,
                      height: 112,
                      borderRadius: 12,
                      objectFit: "cover",
                      flexShrink: 0,
                      border: "1px solid rgba(255,255,255,0.14)",
                      boxShadow: `0 16px 36px rgba(0,0,0,0.65), 0 0 ${haloSpread}px ${accent}${haloAlphaHex}`,
                    }}
                  />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div
                      className="study-serif"
                      style={{
                        fontSize: 28,
                        fontWeight: 700,
                        color: "#f5f1eb",
                        lineHeight: 1.18,
                      }}
                    >
                      {meta.title}
                    </div>
                    <div style={{ fontSize: 18, color: "#9f9690", marginTop: 4 }}>
                      {meta.artist}
                    </div>
                    <div style={{ marginTop: 10 }}>
                      <SpectrumVisualizer
                        spectrum={spectrum}
                        mode="mini-wave"
                        accentColor={accent}
                        width={220}
                        height={30}
                      />
                    </div>
                  </div>
                </div>
                <TrackProgressBar elapsed={elapsed} duration={duration} accent={accent} />
              </div>

              {/* Center Masked Scrolling Lyrics */}
              <div style={{ flex: 1, minHeight: 0 }}>
                <LyricColumn
                  lines={lines}
                  stylePreset={preset}
                  accentColor={accent}
                  showPronunciation={showPronunciation}
                  showWordHighlight={showWordHighlight}
                  karaokeSweep={karaokeSweep}
                  smoothLyricScroll={smoothLyricScroll}
                  lyricWordLift={lyricWordLift}
                  contentScale={effectiveScale}
                />
              </div>

              {/* Fixed Bottom Frosted Study Card */}
              {showNotes ? (
                <div
                  style={{
                    width: "100%",
                    height: 248,
                    overflow: "hidden",
                    padding: "20px 24px",
                    boxSizing: "border-box",
                    borderRadius: 16,
                    backgroundColor: "rgba(18, 15, 24, 0.56)",
                    border: `1px solid color-mix(in oklab, ${accent} 35%, rgba(255,255,255,0.12))`,
                    backdropFilter: "blur(12px)",
                  }}
                >
                  <NoteBody
                    comment={activeNote?.comment}
                    accent={accent}
                    opacity={noteOpacity}
                  />
                </div>
              ) : null}
            </div>
          ) : (
            /* 16:9 Widescreen Layouts: Minimal / Editorial / Classic2 */
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "grid",
                gridTemplateColumns: isEditorial
                  ? "37% 63%"
                  : isClassic
                    ? "38% 62%"
                    : "35% 65%",
                gap: isEditorial || isClassic ? 64 : 80,
                alignItems: "center",
              }}
            >
              {/* Left Column */}
              {isEditorial ? (
                <div
                  style={{
                    height: "100%",
                    borderRight: "1px solid rgba(255,255,255,0.10)",
                    paddingRight: 52,
                    boxSizing: "border-box",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    textAlign: "left",
                  }}
                >
                  {/* Editorial Masthead */}
                  <div style={{ width: "100%" }}>
                    <div
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.22em",
                        textTransform: "uppercase",
                        color: accent,
                        marginBottom: 16,
                      }}
                    >
                      NOW STUDYING · {(meta.language || "EN").toUpperCase()}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                      <Img
                        src={staticFile("track/cover.jpg")}
                        style={{
                          width: 108,
                          height: 108,
                          borderRadius: 10,
                          objectFit: "cover",
                          flexShrink: 0,
                          border: "1px solid rgba(255,255,255,0.14)",
                          boxShadow: `0 18px 40px rgba(0,0,0,0.65), 0 0 ${haloSpread}px ${accent}${haloAlphaHex}`,
                        }}
                      />
                      <div style={{ minWidth: 0 }}>
                        <div
                          className="study-serif"
                          style={{
                            fontSize: 27,
                            fontWeight: 700,
                            color: "#f5f1eb",
                            lineHeight: 1.18,
                          }}
                        >
                          {meta.title}
                        </div>
                        <div style={{ fontSize: 17, color: "#9f9690", marginTop: 6 }}>
                          {meta.artist}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Editorial Spectrum + Progress */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 14, width: 380, maxWidth: "100%" }}>
                    <SpectrumVisualizer
                      spectrum={spectrum}
                      mode="led-bars"
                      accentColor={accent}
                      width={380}
                      height={44}
                    />
                    <TrackProgressBar elapsed={elapsed} duration={duration} accent={accent} width={380} />
                  </div>

                  {/* Editorial Fixed-Height Frosted Pull-Quote Slot */}
                  {showNotes ? (
                    <div
                      style={{
                        width: "100%",
                        maxWidth: 460,
                        height: 256,
                        overflow: "hidden",
                        padding: "18px 22px",
                        boxSizing: "border-box",
                        borderRadius: 10,
                        backgroundColor: "rgba(18, 15, 24, 0.48)",
                        borderLeft: `3px solid ${accent}`,
                        backdropFilter: "blur(8px)",
                      }}
                    >
                      <NoteBody
                        comment={activeNote?.comment}
                        accent={accent}
                        opacity={noteOpacity}
                      />
                    </div>
                  ) : null}
                </div>
              ) : isClassic ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    textAlign: "center",
                    justifyContent: "center",
                    gap: 18,
                  }}
                >
                  {/* Spinning Vinyl LP Disc */}
                  <div
                    style={{
                      position: "relative",
                      width: 310,
                      height: 310,
                      borderRadius: "50%",
                      isolation: "isolate",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transform: `rotate(${((currentTime * 22) % 360).toFixed(2)}deg)`,
                      background:
                        "conic-gradient(from 40deg, rgba(255,255,255,0.12) 0deg, transparent 42deg, rgba(255,255,255,0.03) 140deg, rgba(255,255,255,0.14) 210deg, transparent 260deg, rgba(255,255,255,0.04) 360deg), repeating-radial-gradient(circle at center, #18161d 0px, #18161d 2px, #100e14 3px, #100e14 5px)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      boxShadow: `0 24px 64px rgba(0,0,0,0.75), 0 0 ${haloSpread}px ${accent}${haloAlphaHex}`,
                    }}
                  >
                    <Img
                      src={staticFile("track/cover.jpg")}
                      style={{
                        width: 176,
                        height: 176,
                        borderRadius: "50%",
                        objectFit: "cover",
                        border: `2px solid color-mix(in oklab, ${accent} 65%, transparent)`,
                      }}
                    />
                    <div
                      style={{
                        position: "absolute",
                        width: 18,
                        height: 18,
                        borderRadius: "50%",
                        backgroundColor: "#0c0a0f",
                        border: "2px solid rgba(255,255,255,0.28)",
                      }}
                    />
                  </div>

                  <div>
                    <div
                      className="study-serif"
                      style={{
                        fontSize: 30,
                        fontWeight: 700,
                        letterSpacing: "0.08em",
                        color: "#f5f1eb",
                      }}
                    >
                      {meta.title}
                    </div>
                    <div
                      style={{
                        fontSize: 18,
                        color: "#9c938e",
                        marginTop: 5,
                        letterSpacing: "0.03em",
                      }}
                    >
                      {meta.artist}
                    </div>
                  </div>

                  <SpectrumVisualizer
                    spectrum={spectrum}
                    mode="led-bars"
                    accentColor={accent}
                    width={300}
                    height={40}
                  />

                  <TrackProgressBar elapsed={elapsed} duration={duration} accent={accent} width={300} />

                  {showNotes ? (
                    <div
                      style={{
                        width: "100%",
                        maxWidth: 440,
                        height: 196,
                        overflow: "hidden",
                        paddingTop: 18,
                        boxSizing: "border-box",
                        borderTop: "1px solid rgba(255,255,255,0.16)",
                      }}
                    >
                      <NoteBody
                        comment={activeNote?.comment}
                        accent={accent}
                        opacity={noteOpacity}
                        centered
                        hideTip
                      />
                    </div>
                  ) : null}
                </div>
              ) : (
                /* Minimal Preset */
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    textAlign: "center",
                    justifyContent: "center",
                    gap: 18,
                  }}
                >
                  <Img
                    src={staticFile("track/cover.jpg")}
                    style={{
                      width: 296,
                      height: 296,
                      borderRadius: 14,
                      objectFit: "cover",
                      border: "1px solid rgba(255,255,255,0.12)",
                      boxShadow: `0 24px 60px rgba(0,0,0,0.7), 0 0 ${haloSpread}px ${accent}${haloAlphaHex}`,
                    }}
                  />
                  <div>
                    <div
                      className="study-serif"
                      style={{
                        fontSize: 30,
                        fontWeight: 700,
                        letterSpacing: "0.02em",
                        color: "#f5f1eb",
                      }}
                    >
                      {meta.title}
                    </div>
                    <div style={{ fontSize: 18, color: "#a59c96", marginTop: 6 }}>
                      {meta.artist}
                    </div>
                  </div>

                  <SpectrumVisualizer
                    spectrum={spectrum}
                    mode="mini-wave"
                    accentColor={accent}
                    width={300}
                    height={38}
                  />

                  <TrackProgressBar elapsed={elapsed} duration={duration} accent={accent} width={300} />

                  {showNotes ? (
                    <div
                      style={{
                        width: "100%",
                        maxWidth: 440,
                        height: 224,
                        overflow: "hidden",
                        paddingTop: 18,
                        boxSizing: "border-box",
                        borderTop: "1px solid rgba(255,255,255,0.16)",
                      }}
                    >
                      <NoteBody
                        comment={activeNote?.comment}
                        accent={accent}
                        opacity={noteOpacity}
                      />
                    </div>
                  ) : null}
                </div>
              )}

              {/* Right Column: Masked Scrolling Lyrics */}
              <LyricColumn
                lines={lines}
                stylePreset={preset}
                accentColor={accent}
                showPronunciation={showPronunciation}
                showWordHighlight={showWordHighlight}
                karaokeSweep={karaokeSweep}
                smoothLyricScroll={smoothLyricScroll}
                lyricWordLift={lyricWordLift}
                contentScale={effectiveScale}
              />
            </div>
          )}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
