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
  defaultProps,
}) => {
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
      width: 1920,
      height: 1080,
      props: {
        ...defaultProps,
        timeline,
        spectrum,
        showPronunciation: true,
        showComment: true,
        showWordHighlight: true,
      },
    };
  } catch {
    return {
      fps: 30,
      durationInFrames: 300,
      width: 1920,
      height: 1080,
      props: defaultProps,
    };
  }
};

const COMPOSITIONS: Array<{ id: string; preset: StylePreset }> = [
  { id: "MinimalWave", preset: "minimal" },
  { id: "Editorial", preset: "editorial" },
  { id: "Classic2", preset: "classic2" },
];

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {COMPOSITIONS.map(({ id, preset }) => (
        <Composition
          key={id}
          id={id}
          component={StudyLayout}
          durationInFrames={300}
          fps={30}
          width={1920}
          height={1080}
          defaultProps={{ ...FALLBACK_PROPS, preset }}
          calculateMetadata={loadTrackMetadata}
        />
      ))}
    </>
  );
};
