export type StylePreset = "minimal" | "editorial" | "classic2" | "shorts";

export interface WordTiming {
  text: string;
  start: number;
  end: number;
}

export interface LearningComment {
  tag: string;
  phrase: string;
  meaning: string;
  tip?: string;
}

export interface LyricLine {
  id: number;
  start: number;
  end: number;
  original: string;
  pronunciation?: string;
  translation: string;
  words: WordTiming[];
  whole_line?: boolean;
  comment: LearningComment | null;
  culture_note?: string | null;
}

export interface TrackMeta {
  title: string;
  artist: string;
  language: string;
  duration: number;
  fps: number;
  style?: StylePreset;
  accent_color?: string;
  content_scale?: number;
  karaoke_sweep?: boolean;
  show_word_highlight?: boolean;
  show_pronunciation?: boolean;
  show_comment?: boolean;
  smooth_lyric_scroll?: boolean;
  lyric_word_lift?: boolean;
  show_song_context?: boolean;
  song_context?: { meaning?: string; culture?: string };
}

export interface TimelineData {
  meta: TrackMeta;
  lines: LyricLine[];
}

export interface SpectrumData {
  fps: number;
  num_bars: number;
  total_frames: number;
  frames: number[][];
}

export type CompositionProps = {
  timeline: TimelineData;
  spectrum: SpectrumData;
  preset?: StylePreset;
  showPronunciation?: boolean;
  showComment?: boolean;
  showWordHighlight?: boolean;
  karaokeSweep?: boolean;
  smoothLyricScroll?: boolean;
  lyricWordLift?: boolean;
  showSongContext?: boolean;
};
