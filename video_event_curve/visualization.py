"""Headless plots; rendering is independent of feature extraction."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_event_curve(curves, output_path, title="Video event curve", duration=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    rows = [("raw_event_curve", "native_timestamps", "Cosine dissimilarity"),
            ("normalized_event_curve", "native_timestamps", "Normalized dissimilarity"),
            ("event_curve", "timestamps", "Normalized dissimilarity\n(Hann smoothed)")]
    for ax, (key, time_key, label) in zip(axes, rows):
        ax.plot(curves[time_key], curves[key], linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    if duration is not None:
        axes[-1].set_xlim(0, duration)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (seconds)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_offset_comparison(offset_curves, output_path, title, duration):
    fig, axes = plt.subplots(len(offset_curves), 1, figsize=(12, 3 * len(offset_curves)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for ax, (offset, curves) in zip(axes[:, 0], offset_curves.items()):
        ax.plot(curves["timestamps"], curves["event_curve"], linewidth=1)
        ax.set_ylabel(f"Offset {offset}\nNormalized dissimilarity")
        ax.grid(alpha=0.25)
    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlim(0, duration)
    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
