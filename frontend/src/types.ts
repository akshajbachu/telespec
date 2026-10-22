export type Finding = {
  kind: string; start: number; end: number; magnitude: number; confidence: number;
  summary: string; period: number | null; frequency: number | null;
  evidence: Record<string, unknown>;
};
export type Peak = {
  frequency: number; period: number; power: number; false_alarm_probability: number;
  harmonics: number[];
};
export type Sampling = {
  count: number; start: number; end: number; median_interval: number; jitter_mad: number;
  coverage: number; duplicate_count: number; gaps: unknown[];
};
export type Result = {
  id: string; filename: string; name: string; unit: string; sampling: Sampling;
  timestamps: number[]; values: number[]; frequencies: number[]; power: number[];
  threshold_power: number; peaks: Peak[]; findings: Finding[];
  trend: number[]; seasonal: number[]; remainder: number[];
  anomaly_times: number[]; anomaly_scores: number[];
  metadata: {
    spectrogram: { times: number[]; frequencies: number[]; power: number[][] };
  };
};
