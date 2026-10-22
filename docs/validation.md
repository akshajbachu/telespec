# Validation

The benchmark is executable, deterministic, and stores machine-readable results.

```bash
uv run python benchmarks/run_all.py --quick
uv run python benchmarks/run_all.py --with-nab
```

It measures:

1. generalized Lomb-Scargle versus interpolate-plus-FFT frequency recovery as
   missingness increases;
2. empirical versus nominal false-alarm probability on noise;
3. detection performance by injected effect type and magnitude;
4. selected Numenta Anomaly Benchmark series against expert anomaly windows and
   simple sigma, EWMA, and rolling robust-z baselines.

NAB labels are broad anomaly windows, not causal network diagnoses. Real-data metrics
must therefore be read as detector agreement with those windows, not ground-truth
root-cause accuracy.
