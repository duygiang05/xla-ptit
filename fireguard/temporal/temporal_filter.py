from collections import deque
from fireguard.config import Config
from fireguard.shape.regions import Region
from .features import same_candidate
from .decision import passes_temporal_vote

class TemporalFilter:
    """Simple window voting baseline, NOT validated smoke/fire recognition."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._history = deque(maxlen=cfg.vote_window)

    def update(self, regions: list[Region], frame_idx: int) -> list[Region]:
        self._history.append(list(regions))
        confirmed=[]
        for candidate in regions:
            hits=sum(any(same_candidate(candidate, older, self.cfg) for older in history) for history in self._history)
            if passes_temporal_vote(hits, self.cfg.vote_hits):
                confirmed.append(candidate)
        return confirmed
