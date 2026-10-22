# Methodology

## Sampling comes first

Every ingest returns count, median interval, interval MAD, coverage, duplicates, and
explicit gaps. The primary periodogram never interpolates.

## Generalized Lomb-Scargle

At each angular frequency, telespec solves the weighted least-squares model
`y(t) = a cos(wt) + b sin(wt) + c`. Power is the fractional chi-square reduction from
the floating-mean null. This is the generalized form in Zechmeister and Kuerster
(2009), not the zero-mean classical periodogram.

The global false-alarm threshold is the selected quantile of maximum periodogram
power after repeatedly permuting values over fixed timestamps. This controls the
multiple-frequency search empirically.

## Localization and detection

STFT and Morlet CWT require a regular grid, so that resampling is explicit and rejected
across large gaps. Sliding-window log octave-band powers and remainder variance feed a
robust Minimum Covariance Determinant baseline. Two-sided CUSUM detects persistent
level changes. Findings retain interval, magnitude, confidence, and evidence.

References: Zechmeister & Kuerster, A&A 496 (2009); Baluev, MNRAS 385 (2008);
Page, Biometrika 41 (1954); Rousseeuw & Van Driessen, Technometrics 41 (1999).
