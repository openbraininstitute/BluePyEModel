"""Convert frozen text noise assets to exactly reversible numeric archives."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy

from bluepyemodel.ecode.noise_data import decode_noise_data
from bluepyemodel.ecode.noise_data import encode_noise_data


def main():
    """Convert one asset, checking both numeric and original-text round trips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    series = numpy.loadtxt(io.BytesIO(raw))
    encoded = encode_noise_data(series)
    restored = decode_noise_data(encoded)
    if restored.tobytes() != series.tobytes():
        raise ValueError("Noise conversion changed sample bits.")
    text = io.BytesIO()
    numpy.savetxt(text, restored)
    if text.getvalue() != raw:
        raise ValueError("Source text is not reproducible in the original numeric format.")
    source_hash = hashlib.sha256(raw).hexdigest()
    encoded["source_sha256"] = numpy.asarray(source_hash)
    with args.destination.open("xb") as output:
        numpy.savez_compressed(output, **encoded)
    print(json.dumps({"source_sha256": source_hash, "rows": len(series)}))


if __name__ == "__main__":
    main()
