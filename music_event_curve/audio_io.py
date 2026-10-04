from pathlib import Path
import av
import numpy as np


def load_audio(path, sample_rate=24000):
    """Decode first audio stream, arithmetic channel mean, resample with FFmpeg.

    Preserve the stream's starting PTS. Reject discontinuities rather than
    concatenating gaps away. Codec padding is trimmed to the declared duration.
    """
    parts, first_time, expected_time = [], None, None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("Input has no audio track")
        stream = container.streams.audio[0]
        original_rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout,
                                     rate=sample_rate)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Audio frame lacks presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if first_time is None:
                first_time = timestamp
            if expected_time is not None and abs(timestamp - expected_time) > 2 / original_rate:
                raise ValueError("Discontinuous audio timestamps; alignment would be ambiguous")
            expected_time = timestamp + frame.samples / frame.sample_rate
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().mean(axis=0))
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().mean(axis=0))
    if not parts:
        raise ValueError("No audio samples decoded")
    waveform = np.concatenate(parts).astype(np.float32)
    if duration is not None:
        waveform = waveform[:round(duration * sample_rate)]
    if len(waveform) <= 1024 or not np.isfinite(waveform).all():
        raise ValueError("Audio must be finite and longer than the STFT reflection padding")
    return waveform, {"audio_path": str(Path(path).resolve()), "original_sample_rate": original_rate,
                      "original_channels": channels, "sample_rate": sample_rate,
                      "audio_start_pts_seconds": first_time, "num_samples": len(waveform),
                      "duration": len(waveform) / sample_rate,
                      "downmix": "arithmetic channel mean", "resampler": "PyAV/FFmpeg"}
