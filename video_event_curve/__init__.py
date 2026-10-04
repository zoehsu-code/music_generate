"""Video-only V2M-ZERO event-curve baseline."""


def extract_video_event_curve(*args, **kwargs):
    """See pipeline.extract_video_event_curve for the full API."""
    from .pipeline import extract_video_event_curve as extract
    return extract(*args, **kwargs)
