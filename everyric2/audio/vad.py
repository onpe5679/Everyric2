"""Vocal Activity Detection for post-processing subtitle timing.

Analyzes separated vocals audio to detect actual voice regions,
which can be used to adjust CTC alignment results.
"""

import logging
from dataclasses import dataclass

import numpy as np

from everyric2.audio.loader import AudioData

logger = logging.getLogger(__name__)


@dataclass
class VocalRegion:
    start: float
    end: float
    energy: float


@dataclass
class VADResult:
    regions: list[VocalRegion]
    sample_rate: int
    frame_duration: float
    energy_threshold: float
    total_duration: float


class VocalActivityDetector:
    def __init__(
        self,
        frame_duration: float = 0.02,
        energy_threshold_percentile: float = 40.0,
        min_region_duration: float = 0.2,
        merge_gap: float = 0.3,
    ):
        self.frame_duration = frame_duration
        self.energy_threshold_percentile = energy_threshold_percentile
        self.min_region_duration = min_region_duration
        self.merge_gap = merge_gap

    def detect(self, audio: AudioData) -> VADResult:
        waveform = audio.waveform
        sr = audio.sample_rate

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        frame_samples = int(self.frame_duration * sr)
        num_frames = len(waveform) // frame_samples

        energies = []
        for i in range(num_frames):
            start_idx = i * frame_samples
            end_idx = start_idx + frame_samples
            frame = waveform[start_idx:end_idx]
            energy = np.sqrt(np.mean(frame**2))
            energies.append(energy)

        energies = np.array(energies)

        non_zero_energies = energies[energies > 0]
        if len(non_zero_energies) > 0:
            threshold = np.percentile(non_zero_energies, self.energy_threshold_percentile)
        else:
            threshold = 0.001

        is_voice = energies > threshold

        regions = []
        in_region = False
        region_start = 0
        region_energies = []

        for i, voiced in enumerate(is_voice):
            if voiced and not in_region:
                in_region = True
                region_start = i
                region_energies = [energies[i]]
            elif voiced and in_region:
                region_energies.append(energies[i])
            elif not voiced and in_region:
                in_region = False
                start_time = region_start * self.frame_duration
                end_time = i * self.frame_duration
                avg_energy = np.mean(region_energies)
                regions.append(VocalRegion(start=start_time, end=end_time, energy=avg_energy))

        if in_region:
            start_time = region_start * self.frame_duration
            end_time = num_frames * self.frame_duration
            avg_energy = np.mean(region_energies)
            regions.append(VocalRegion(start=start_time, end=end_time, energy=avg_energy))

        merged = self._merge_close_regions(regions)
        filtered = [r for r in merged if (r.end - r.start) >= self.min_region_duration]

        logger.info(f"VAD: {len(filtered)} vocal regions detected (threshold={threshold:.4f})")

        return VADResult(
            regions=filtered,
            sample_rate=sr,
            frame_duration=self.frame_duration,
            energy_threshold=threshold,
            total_duration=audio.duration,
        )

    def _merge_close_regions(self, regions: list[VocalRegion]) -> list[VocalRegion]:
        if not regions:
            return []

        merged = [regions[0]]
        for region in regions[1:]:
            prev = merged[-1]
            gap = region.start - prev.end

            if gap <= self.merge_gap:
                merged[-1] = VocalRegion(
                    start=prev.start,
                    end=region.end,
                    energy=(prev.energy + region.energy) / 2,
                )
            else:
                merged.append(region)

        return merged

    def get_vocal_at_time(self, vad_result: VADResult, time: float) -> VocalRegion | None:
        for region in vad_result.regions:
            if region.start <= time <= region.end:
                return region
        return None

    def find_nearest_vocal_end(
        self, vad_result: VADResult, time: float, max_search: float = 2.0
    ) -> float | None:
        for region in vad_result.regions:
            if region.start <= time <= region.end:
                return region.end
            if region.start > time and region.start - time <= max_search:
                return None
        return None

    def is_silence_at_time(self, vad_result: VADResult, time: float) -> bool:
        return self.get_vocal_at_time(vad_result, time) is None
