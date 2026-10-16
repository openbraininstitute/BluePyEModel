"""Replay physical currents without normalization or additional holding."""

import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus


class RecordedCurrent(BPEM_stimulus):
    """Replay a complete current waveform in nA at its original times in ms.

    Delay and duration define the feature window, not a shift of the waveform.
    Holding current and rheobase parameters never alter these absolute samples.
    """

    name = "RecordedCurrent"

    def __init__(self, location, **kwargs):
        """Validate and own the physical waveform and its sampling clock."""
        if kwargs.get("waveform_includes_holding", True) is not True:
            raise ValueError("RecordedCurrent requires physical current including holding.")
        if kwargs.get("waveform_unit", "nA") != "nA" or kwargs.get("time_unit", "ms") != "ms":
            raise ValueError("RecordedCurrent expects explicit nA and ms units.")
        if kwargs.get("data_filepath") is not None:
            raise ValueError("A recorded waveform cannot be combined with data_filepath.")
        current = numpy.asarray(kwargs.get("waveform"), dtype=numpy.float64)
        if current.ndim != 1 or len(current) < 2 or not numpy.isfinite(current).all():
            raise ValueError("Recorded waveform must be a finite vector with at least two samples.")
        step = float(kwargs.get("dt", 0.1))
        if not numpy.isfinite(step) or step <= 0:
            raise ValueError("Recorded sampling dt must be finite and positive.")
        times = kwargs.get("time_series")
        time = (
            numpy.arange(len(current), dtype=numpy.float64) * step
            if times is None
            else numpy.asarray(times, dtype=numpy.float64)
        )
        if (
            time.shape != current.shape
            or not numpy.isfinite(time).all()
            or time[0] < 0
            or numpy.any(numpy.diff(time) <= 0)
        ):
            raise ValueError(
                "Recorded times must align with current and increase from non-negative time."
            )
        duration = kwargs.get("totduration")
        if duration is not None and float(duration) != float(time[-1]):
            raise ValueError("Recorded total duration must match the last timestamp.")
        self.delay = float(kwargs.get("delay", time[0]))
        window = kwargs.get("duration")
        self._stim_end = float(time[-1]) if window is None else self.delay + float(window)
        if (
            not numpy.isfinite([self.delay, self._stim_end]).all()
            or not time[0] <= self.delay < self._stim_end <= time[-1]
        ):
            raise ValueError("Recorded feature interval is outside the waveform.")
        self.amp = (
            float(kwargs["amp"]) if kwargs.get("amp") is not None else float(numpy.std(current))
        )
        if not numpy.isfinite(self.amp):
            raise ValueError("Recorded amplitude metadata must be finite.")
        self.time_series = time.copy()
        self.current_series = current.copy()
        self.holding_current = kwargs.get("holding_current")
        self.threshold_current = None
        self.amp_rel = None
        super().__init__(location=location)

    @property
    def total_duration(self):
        """Return the actual last recorded timestamp."""
        return float(self.time_series[-1])

    @property
    def stim_start(self):
        """Return the feature-window start without shifting stimulus samples."""
        return self.delay

    @property
    def stim_end(self):
        """Return the feature-window end."""
        return self._stim_end

    @property
    def amplitude(self):
        """Return selection metadata, which is never applied to the waveform."""
        return self.amp

    def generate(self, dt=0.1):
        """Return original samples; the interface dt argument never resamples."""
        del dt
        return self.time_series.copy(), self.current_series.copy()
