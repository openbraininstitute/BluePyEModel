"""Lossless encoding of frozen noise currents and timestamp bit patterns."""

from importlib import resources

import numpy

NOISE_NAMES = ("WhiteNoise", "NoiseOU3", "SubWhiteNoise")
FORMAT_VERSION = 1


def _validate_series(series):
    """Validate finite two-column data with strictly increasing timestamps."""
    series = numpy.asarray(series, dtype=numpy.float64)
    if series.ndim != 2 or series.shape[1] != 2 or len(series) < 2:
        raise ValueError("Noise data must have at least two time/current rows.")
    if not numpy.isfinite(series).all() or numpy.any(numpy.diff(series[:, 0]) <= 0):
        raise ValueError("Noise data must be finite with increasing timestamps.")
    return series


def encode_noise_data(series):
    """Encode float64 samples without changing any timestamp or current bits."""
    series = _validate_series(series)
    values, index = numpy.unique(series[:, 1].view(numpy.uint64), return_inverse=True)
    if len(values) > numpy.iinfo(numpy.uint16).max + 1:
        raise ValueError("Noise current dictionary exceeds uint16 capacity.")
    return {
        "format_version": numpy.uint16(FORMAT_VERSION),
        "values": values.view(numpy.float64).astype("<f8"),
        "index": index.astype("<u2"),
        "time_deltas": numpy.diff(series[:, 0].view(numpy.uint64), prepend=numpy.uint64(0)).astype(
            "<u8"
        ),
    }


def decode_noise_data(data):
    """Decode and validate a numeric archive, preserving the original bits."""
    version = numpy.asarray(data["format_version"])
    if version.ndim != 0 or version.dtype.kind not in "iu" or version != FORMAT_VERSION:
        raise ValueError("Unsupported noise data format version.")
    values = numpy.asarray(data["values"])
    index = numpy.asarray(data["index"])
    deltas = numpy.asarray(data["time_deltas"])
    for array, kind, size in ((values, "f", 8), (index, "u", 2), (deltas, "u", 8)):
        if array.ndim != 1 or array.dtype.kind != kind or array.dtype.itemsize != size:
            raise ValueError("Invalid noise data array shape or dtype.")
    if len(index) < 2 or len(index) != len(deltas) or values.size == 0:
        raise ValueError("Invalid noise data array lengths.")
    if numpy.max(index) >= len(values):
        raise ValueError("Noise data index is outside the current dictionary.")
    time = numpy.cumsum(deltas, dtype=numpy.uint64).view(numpy.float64)
    return _validate_series(numpy.column_stack((time, values[index])))


def load_noise_data(name):
    """Load a bundled default using a binary, filesystem-independent resource."""
    if name not in NOISE_NAMES:
        raise ValueError(f"Unknown bundled noise protocol: {name}")
    resource = resources.files("bluepyemodel") / "ecode" / "data" / f"{name}.npz"
    with resource.open("rb") as stream, numpy.load(stream, allow_pickle=False) as data:
        return decode_noise_data(data)
