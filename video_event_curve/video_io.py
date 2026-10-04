"""Stream every frame with PyAV; keep presentation timestamps (including VFR)."""
from __future__ import annotations

from pathlib import Path
import av
import numpy as np


class VideoReader:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.metadata = {"video_path": str(self.path)}
        self.timestamps = None

    def extract_frames(self):
        """Yield (RGB PIL image, elapsed seconds); metadata finalizes at EOF."""
        with av.open(str(self.path)) as container:
            if not container.streams.video:
                raise ValueError("Input has no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            reported_duration = (float(stream.duration * stream.time_base)
                                 if stream.duration is not None else None)
            self.metadata.update(fps=fps, reported_num_frames=stream.frames,
                                 reported_duration=reported_duration,
                                 video_stream_index=stream.index)
            times, origin, last_duration = [], None, None
            for frame in container.decode(stream):
                # Do not silently replace missing PTS with a guessed constant FPS:
                # that would corrupt VFR alignment. Fail with a useful diagnostic.
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Video frame lacks PTS; remux with valid timestamps first")
                timestamp = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = timestamp
                timestamp -= origin
                if not np.isfinite(timestamp) or (times and timestamp <= times[-1]):
                    raise ValueError("Video PTS must be finite and strictly increasing")
                times.append(timestamp)
                duration = getattr(frame, "duration", 0)
                last_duration = float(duration * frame.time_base) if duration else None
                yield frame.to_image().convert("RGB"), timestamp
            if not times:
                raise ValueError("Video contains no decodable frames")
            # Implementation choice: observed timeline including the last frame.
            # Prefer its packet duration; otherwise estimate only that final span.
            duration_source = "last_frame_duration"
            if not last_duration or last_duration <= 0:
                last_duration = (times[-1] - times[-2] if len(times) > 1
                                 else (1 / fps if fps and fps > 0 else 0.0))
                duration_source = "estimated_last_frame_interval"
            self.timestamps = np.asarray(times, dtype=np.float64)
            self.metadata.update(num_frames=len(times), duration=times[-1] + last_duration,
                                 duration_source=duration_source,
                                 timestamp_source="presentation_timestamps",
                                 first_frame_pts_seconds=origin,
                                 timestamp_origin="first_decoded_frame",
                                 last_frame_time=times[-1])


def load_video(path):
    return VideoReader(path)
