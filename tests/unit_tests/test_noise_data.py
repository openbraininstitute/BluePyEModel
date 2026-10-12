"""Regression checks for the lossless bundled noise encoding."""

import hashlib
import io
from importlib import resources

import numpy
import pytest

from bluepyemodel.ecode import eCodes
from bluepyemodel.ecode.noise_data import decode_noise_data
from bluepyemodel.ecode.noise_data import encode_noise_data
from bluepyemodel.ecode.noise_data import load_noise_data

GOLDEN = {
    "SubWhiteNoise": (
        50999,
        "a8962ffafd9c17525f4f4f7c9c45ee568249265f703ad51b6f15798db8be1ca8",
        "b41291a615ff9f1eb1a0b3ac6f8467c478546d706aab36849a222837f1361ac3",
    ),
    "WhiteNoise": (
        601000,
        "1f1186c4097092240f64fd130a0945bbe897cffc8cae7e8729c205bb8f581f39",
        "9f27cd961456748f0766ae5e601206182f5ddd3a262a70f0996b70faad36176e",
    ),
    "NoiseOU3": (
        601000,
        "1f1186c4097092240f64fd130a0945bbe897cffc8cae7e8729c205bb8f581f39",
        "c0b744622d71b8608e4828322502aa6e7be2f9148bfddb586a12365ad91c059d",
    ),
}


@pytest.mark.parametrize("name", GOLDEN)
def test_default_arrays_are_bit_identical(name):
    series = load_noise_data(name)
    length, time_hash, current_hash = GOLDEN[name]
    assert series.shape == (length, 2)
    assert series.dtype == numpy.float64
    for column, expected in enumerate((time_hash, current_hash)):
        canonical = numpy.asarray(series[:, column], dtype="<f8")
        assert hashlib.sha256(canonical.tobytes()).hexdigest() == expected


@pytest.mark.parametrize("name", GOLDEN)
def test_original_text_hash_can_be_recreated(name):
    resource = resources.files("bluepyemodel") / "ecode" / "data" / f"{name}.npz"
    with resource.open("rb") as stream, numpy.load(stream, allow_pickle=False) as data:
        expected = str(data["source_sha256"])
    text = io.BytesIO()
    numpy.savetxt(text, load_noise_data(name))
    assert hashlib.sha256(text.getvalue()).hexdigest() == expected


def test_subwhitenoise_threshold_scaling_is_unchanged():
    stimulus = eCodes["subwhitenoise"](location=None, thresh_perc=150.0, holding_current=-0.1)
    stimulus.threshold_current = 0.3
    _, current = stimulus.generate()
    expected = -0.1 + (0.3 * 1.5) * load_noise_data("SubWhiteNoise")[:, 1]
    assert current.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", GOLDEN)
@pytest.mark.parametrize("holding", [None, 0.0, -0.1])
def test_generated_current_matches_original_formula(name, holding):
    kwargs = {"holding_current": holding}
    if name == "SubWhiteNoise":
        kwargs.update(amp=0.2, thresh_perc=None)
    else:
        kwargs["mu"] = 0.375
    stimulus = eCodes[name.lower()](location=None, **kwargs)
    series = load_noise_data(name)
    time, current = stimulus.generate()
    base = 0.0 if holding is None else holding
    expected = (
        base + 0.2 * series[:, 1]
        if name == "SubWhiteNoise"
        else base + series[:, 1] * (0.375 / 2.0) + 0.375
    )
    assert time.tobytes() == series[:, 0].tobytes()
    assert current.tobytes() == expected.tobytes()
    assert stimulus.stim_start == 0.0
    assert stimulus.stim_end == time[-1]
    assert stimulus.total_duration == time[-1]
    with pytest.raises(ValueError, match="dt has to be"):
        stimulus.generate(dt=0.2)


@pytest.mark.parametrize("name", GOLDEN)
def test_custom_text_still_supported(name, tmp_path):
    path = tmp_path / "custom.txt"
    series = numpy.column_stack(([0.0, 0.1, 0.2], [1.0, -1.0, 0.5]))
    numpy.savetxt(path, series)
    kwargs = {"data_filepath": path, "amp": 0.2, "mu": 0.375}
    stimulus = eCodes[name.lower()](location=None, **kwargs)
    assert stimulus.time_series.tobytes() == series[:, 0].tobytes()
    assert stimulus.current_series.tobytes() == series[:, 1].tobytes()


def test_encoding_preserves_signed_zero_and_endianness():
    series = numpy.column_stack(([0.0, 0.1, 0.2], [-0.0, 0.0, 0.2]))
    packed = encode_noise_data(series)
    packed["time_deltas"] = packed["time_deltas"].astype(">u8")
    packed["values"] = packed["values"].astype(">f8")
    assert decode_noise_data(packed).tobytes() == series.tobytes()
    buffer = io.BytesIO()
    numpy.savez_compressed(buffer, **packed)
    with numpy.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as saved:
        assert decode_noise_data(saved).tobytes() == series.tobytes()


@pytest.mark.parametrize(
    "series",
    [numpy.zeros(3), numpy.zeros((0, 2)), [[0, 1], [0, 2]], [[0, 1], [1, numpy.nan]]],
)
def test_encoder_rejects_invalid_series(series):
    with pytest.raises(ValueError):
        encode_noise_data(series)


def test_encoder_rejects_dictionary_overflow():
    series = numpy.column_stack((numpy.arange(65537), numpy.arange(65537)))
    with pytest.raises(ValueError, match="dictionary"):
        encode_noise_data(series)


@pytest.mark.parametrize(
    "key,value",
    [
        ("format_version", numpy.uint16(2)),
        ("format_version", numpy.array([1])),
        ("values", numpy.array([1], dtype=numpy.float32)),
        ("values", numpy.array([], dtype=numpy.float64)),
        ("values", numpy.array([[1]], dtype=numpy.float64)),
        ("index", numpy.array([2, 0], dtype=numpy.uint16)),
        ("index", numpy.array([0, 0], dtype=numpy.int16)),
        ("time_deltas", numpy.array([0], dtype=numpy.uint64)),
        ("time_deltas", numpy.array([0, 0], dtype=numpy.uint64)),
    ],
)
def test_decoder_rejects_invalid_data(key, value):
    packed = encode_noise_data(numpy.array([[0.0, 1.0], [0.1, 2.0]]))
    packed[key] = value
    with pytest.raises(ValueError):
        decode_noise_data(packed)


def test_subwhitenoise_requires_amplitude():
    with pytest.raises(TypeError, match="cannot be both None"):
        eCodes["subwhitenoise"](location=None, amp=None, thresh_perc=None)


def test_unknown_default_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        load_noise_data("../custom")
