
"""TV4: Temporal tracking, voting and bounding-box suppression.

Pipeline API:
    TemporalFilter(cfg).update(regions, frame_idx) -> list[Region]

Modes:
    vote   : temporal voting only
    hybrid : temporal voting + shape evidence

Bounding-box filtering:
    FIRE  : overlap 25%, IoU 50%
    SMOKE : overlap 40%, IoU 50% (experimental)

Region.score is not modified.
"""

from collections import deque

from fireguard.config import Config
from fireguard.shape.regions import Region
from fireguard.shape.tracking import RegionTracker, Track

from .decision import (
    ShapeThresholds,
    passes_temporal_vote,
    passes_shape_evidence,
)


# ============================================================
# Bounding-box filtering
# ============================================================

def suppress_overlapping_boxes(
    regions: list[Region],
    kind: str,
    containment_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> list[Region]:
    """Remove redundant overlapping boxes of a specific kind.

    Rules:
    - Process larger bounding boxes first.
    - Remove a smaller candidate when its overlap ratio
      or IoU exceeds the selected threshold.
    - Keep non-overlapping boxes.
    - Preserve boxes belonging to other region kinds.
    - Preserve original order in the output.

    This performs box suppression, not region merging.
    """

    if not 0 <= containment_threshold <= 1:
        raise ValueError(
            "containment_threshold must be between 0 and 1"
        )

    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            "iou_threshold must be between 0 and 1"
        )

    def area(box):
        return max(0, box[2]) * max(0, box[3])

    def intersection(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])

        x2 = min(a[0] + a[2], b[0] + b[2])
        y2 = min(a[1] + a[3], b[1] + b[3])

        return (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

    # Collect only valid boxes of the selected kind.
    indexed_regions = [
        (index, region)
        for index, region in enumerate(regions)
        if region.kind == kind and area(region.bbox) > 0
    ]

    # Prioritize larger bounding boxes.
    indexed_regions.sort(
        key=lambda item: -area(item[1].bbox)
    )

    kept = []

    for index, candidate in indexed_regions:
        candidate_area = area(candidate.bbox)
        redundant = False

        for _, existing in kept:
            existing_area = area(existing.bbox)

            inter = intersection(
                candidate.bbox,
                existing.bbox,
            )

            if inter == 0:
                continue

            union = (
                candidate_area
                + existing_area
                - inter
            )

            iou = (
                inter / union
                if union > 0
                else 0.0
            )

            overlap_ratio = inter / candidate_area

            if (
                overlap_ratio >= containment_threshold
                or iou >= iou_threshold
            ):
                redundant = True
                break

        if not redundant:
            kept.append((index, candidate))

    kept_indices = {
        index
        for index, _ in kept
    }

    # Preserve original order and all other region kinds.
    return [
        region
        for index, region in enumerate(regions)
        if (
            region.kind != kind
            or area(region.bbox) == 0
            or index in kept_indices
        )
    ]


def suppress_fire_boxes(
    regions: list[Region],
    containment_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> list[Region]:
    """Compatibility wrapper for the existing FIRE filter."""

    return suppress_overlapping_boxes(
        regions,
        kind="fire",
        containment_threshold=containment_threshold,
        iou_threshold=iou_threshold,
    )


# ============================================================
# TV4 Temporal Filter
# ============================================================

class TemporalFilter:
    """Confirm FIRE/SMOKE regions using TV3 RegionTracker.

    - Reuse TV3 RegionTracker and Track.
    - Count real observations within a K/N voting window.
    - Support vote and hybrid decision modes.
    - Keep debugging information in last_decisions.
    - Filter redundant FIRE and SMOKE bounding boxes.
    """

    def __init__(
        self,
        cfg: Config,
        decision_mode: str = "vote",
        shape_thresholds: ShapeThresholds | None = None,
        max_missed: int = 2,
    ):
        if (
            cfg.vote_window < 1
            or not 1 <= cfg.vote_hits <= cfg.vote_window
        ):
            raise ValueError(
                "Can 1 <= vote_hits <= vote_window"
            )

        if decision_mode not in ("vote", "hybrid"):
            raise ValueError(
                "decision_mode must be 'vote' or 'hybrid'"
            )

        if max_missed < 0:
            raise ValueError(
                "max_missed must not be negative"
            )

        self.cfg = cfg
        self.decision_mode = decision_mode

        self.shape_thresholds = (
            shape_thresholds or ShapeThresholds()
        )

        self.max_missed = max_missed

        self.tracker = RegionTracker(
            cfg,
            history=cfg.vote_window,
            max_missed=max_missed,
        )

        self._last_frame_idx = -1

        self.last_decisions: list[dict] = []

    def update(
        self,
        regions: list[Region],
        frame_idx: int,
    ) -> list[Region]:
        """Process regions for one frame.

        The frame index must increase.

        If some frames are skipped, update the tracker
        with empty observations for the missing frames.
        """

        # --------------------------------------------------------
        # Validate frame order
        # --------------------------------------------------------

        if frame_idx <= self._last_frame_idx:
            raise ValueError(
                "frame_idx must increase"
            )

        # --------------------------------------------------------
        # Handle skipped frames
        # --------------------------------------------------------

        if self._last_frame_idx >= 0:
            gap = frame_idx - self._last_frame_idx - 1

            if gap > self.max_missed:
                # Existing tracks cannot survive such a gap.
                self.tracker.tracks.clear()
            else:
                for missed_frame in range(
                    self._last_frame_idx + 1,
                    frame_idx,
                ):
                    self.tracker.update([], missed_frame)

        self._last_frame_idx = frame_idx

        self.last_decisions = []

        # --------------------------------------------------------
        # TV3 RegionTracker
        # --------------------------------------------------------

        assignments = self.tracker.update(
            regions,
            frame_idx,
        )

        result = []

        earliest = (
            frame_idx
            - self.cfg.vote_window
            + 1
        )

        # --------------------------------------------------------
        # Temporal Voting / Hybrid Decision
        # --------------------------------------------------------

        for track_id, region in assignments:
            track = self.tracker.get(track_id)

            assert track is not None

            # Count distinct frames with real observations.
            hits = len({
                idx
                for idx, _ in track.history
                if earliest <= idx <= frame_idx
            })

            # Reuse TV3's geometric change calculations.
            recent = deque(
                (
                    (idx, item)
                    for idx, item in track.history
                    if earliest <= idx <= frame_idx
                ),
                maxlen=self.cfg.vote_window,
            )

            shape = Track(
                track.track_id,
                track.kind,
                recent,
            ).shape_change()

            vote_ok = passes_temporal_vote(
                hits,
                self.cfg.vote_hits,
            )

            shape_ok = passes_shape_evidence(
                region.kind,
                shape,
                self.shape_thresholds,
            )

            accepted = (
                vote_ok
                and (
                    self.decision_mode == "vote"
                    or shape_ok
                )
            )

            if accepted:
                result.append(region)

            self.last_decisions.append({
                "frame": frame_idx,
                "track_id": track_id,
                "kind": region.kind,
                "bbox": region.bbox,
                "hits": hits,
                "window": self.cfg.vote_window,
                "vote_pass": vote_ok,
                "shape_pass": shape_ok,
                "confirmed": accepted,
                "mode": self.decision_mode,
                **shape,
            })

        # --------------------------------------------------------
        # FIRE: preserve the validated V1 thresholds
        # --------------------------------------------------------

        result = suppress_overlapping_boxes(
            result,
            kind="fire",
            containment_threshold=0.25,
            iou_threshold=0.50,
        )

        # --------------------------------------------------------
        # SMOKE: experimental overlap filtering
        # --------------------------------------------------------

        result = suppress_overlapping_boxes(
            result,
            kind="smoke",
            containment_threshold=0.40,
            iou_threshold=0.50,
        )

        return result
