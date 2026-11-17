import numpy as np
from video_event_curve.event_curve import standardize_curve, hann_smooth


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float('nan')
    x, y = x - x.mean(), y - y.mean()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float('nan')
    return float(np.clip(np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y)), -1, 1))


def compare(vraw, vt, mraw, mt, rate=25, window=31, max_lag_seconds=2):
    for x, t in ((vraw, vt), (mraw, mt)):
        if len(x) != len(t) or len(t) < 3 or not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Invalid curve or time axis')
    if rate <= 0 or max_lag_seconds < 0:
        raise ValueError('Invalid rate or lag range')
    start, end = max(vt[0], mt[0]), min(vt[-1], mt[-1])
    grid = np.arange(np.ceil(start*rate), np.floor(end*rate)+1) / rate
    if len(grid) < 3:
        raise ValueError('Insufficient common time support')
    v = np.interp(grid, vt, standardize_curve(vraw))
    m = np.interp(grid, mt, standardize_curve(mraw))
    vs, ms = hann_smooth(v, window), hann_smooth(m, window)
    r = pearson(vs, ms)
    quality = 'ok'
    if np.std(vraw) < 1e-8 or np.std(mraw) < 1e-8 or not np.isfinite(r):
        quality = 'near_constant'; r = float('nan')
    elif grid[-1]-grid[0] < max(5, 3*(window-1)/rate):
        quality = 'short_support'
    # Identical video anchors at every lag; no circular wrap or extrapolation.
    k = int(round(max_lag_seconds*rate))
    lags = np.arange(-k, k+1)
    scores = np.full(len(lags), np.nan)
    if len(grid)-2*k >= max(3, int(5*rate)) and quality != 'near_constant':
        anchor = np.arange(k, len(grid)-k)
        scores = np.array([pearson(vs[anchor], ms[anchor+j]) for j in lags])
    best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
    return dict(timestamps=grid, video_standardized=v, music_standardized=m,
                video=vs, music=ms, lag_seconds=lags/rate, lag_correlations=scores), dict(
        pearson_zero_lag=r, pearson_unsmoothed=pearson(v,m) if quality != 'near_constant' else float('nan'),
        pearson_smooth_2_4s=pearson(hann_smooth(v,61),hann_smooth(m,61)) if quality != 'near_constant' else float('nan'),
        pearson_best_lag=float(scores[best]) if best is not None else float('nan'),
        pearson_zero_lag_fixed_support=float(scores[k]),
        best_lag_seconds=float(lags[best]/rate) if best is not None else float('nan'),
        valid_duration_seconds=float(grid[-1]-grid[0]), quality_flag=quality)
