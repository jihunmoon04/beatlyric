import React from "react";
import { CalculateMetadataFunction, Composition, staticFile } from "remotion";
import { StudyLayout } from "./templates/StudyLayout";
import type { CompositionProps, SpectrumData, StylePreset, TimelineData } from "./types";

const FALLBACK_PROPS: CompositionProps = {
  timeline: {
    meta: {
      title: "DEMO TRACK",
      artist: "Sample Artist",
      language: "en",
      duration: 10,
      fps: 30,
      style: "minimal",
    },
    lines: [],
  },
  spectrum: {
    fps: 30,
    num_bars: 28,
    total_frames: 300,
    frames: [],
  },
  preset: "minimal",
};

const loadTrackMetadata: CalculateMetadataFunction<CompositionProps> = async ({
  props,
  defaultProps,
}) => {
  const preset = props.preset ?? defaultProps.preset ?? "minimal";
  const isShorts = preset === "shorts";
  const width = isShorts ? 1080 : 1920;
  const height = isShorts ? 1920 : 1080;

  try {
    const [tlRes, spRes] = await Promise.all([
      fetch(staticFile("track/timeline.json")),
      fetch(staticFile("track/spectrum.json")),
    ]);
    const timeline = (await tlRes.json()) as TimelineData;
    const spectrum = (await spRes.json()) as SpectrumData;
    const fps = timeline.meta.fps || 30;
    const durationInFrames = Math.max(1, Math.ceil(timeline.meta.duration * fps));
    return {
      fps,
      durationInFrames,
      width,
      height,
      props: {
        ...defaultProps,
        ...props,
        timeline,
        spectrum,
        showPronunciation: props.showPronunciation ?? timeline.meta.show_pronunciation ?? true,
        showComment: props.showComment ?? timeline.meta.show_comment ?? true,
        showWordHighlight: props.showWordHighlight ?? timeline.meta.show_word_highlight ?? true,
        smoothLyricScroll: props.smoothLyricScroll ?? timeline.meta.smooth_lyric_scroll ?? true,
        lyricWordLift: props.lyricWordLift ?? timeline.meta.lyric_word_lift ?? true,
        showSongContext: props.showSongContext ?? timeline.meta.show_song_context ?? true,
        karaokeSweep: props.karaokeSweep ?? timeline.meta.karaoke_sweep ?? true,
      },
    };
  } catch {
    return {
      fps: 30,
      durationInFrames: 300,
      width,
      height,
      props: defaultProps,
    };
  }
};

const COMPOSITIONS: Array<{ id: string; preset: StylePreset; width: number; height: number }> = [
  { id: "MinimalWave", preset: "minimal", width: 1920, height: 1080 },
  { id: "Editorial", preset: "editorial", width: 1920, height: 1080 },
  { id: "Classic2", preset: "classic2", width: 1920, height: 1080 },
  { id: "ShortsVertical", preset: "shorts", width: 1080, height: 1920 },
];

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {COMPOSITIONS.map(({ id, preset, width, height }) => (
        <Composition
          key={id}
          id={id}
          component={StudyLayout}
          durationInFrames={300}
          fps={30}
          width={width}
          height={height}
          defaultProps={{ ...FALLBACK_PROPS, preset }}
          calculateMetadata={loadTrackMetadata}
        />
      ))}
    </>
  );
};
