"""32-bit integer arithmetic reference coder with conditional Gaussian CDFs.

CPU reference bitstream. The benchmark CUDA path uses HAC's original codec.
"""
import math
import struct
import numpy as np
from scipy.special import ndtr

TOTAL = 1 << 20
MASK = (1 << 32)-1
HALF, QUARTER, THREE_QUARTERS = 1 << 31, 1 << 30, 3 << 30


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.value = self.bits = 0

    def write(self, bit):
        self.value = (self.value << 1) | int(bit)
        self.bits += 1
        if self.bits == 8:
            self.data.append(self.value)
            self.value = self.bits = 0

    def finish(self):
        if self.bits:
            self.data.append(self.value << (8-self.bits))
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read(self):
        offset = self.offset
        self.offset += 1
        return ((self.data[offset//8] >> (7-offset % 8)) & 1) if offset//8 < len(self.data) else 0


def _boundary(symbol, minimum, maximum, mean, std, step):
    if symbol <= minimum:
        return 0
    if symbol > maximum:
        return TOTAL
    alphabet = maximum-minimum+1
    p = float(ndtr(((symbol-0.5)*step-mean)/max(std, 1e-6)))
    return min(TOTAL, max(0, int(p*(TOTAL-alphabet))+symbol-minimum))


def _contexts(means, stds, steps):
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64).reshape(-1)
                                    for a in (means, stds, steps)])
    if (not all(np.isfinite(a).all() for a in arrays)
            or (arrays[1] <= 0).any() or (arrays[2] <= 0).any()):
        raise ValueError("Invalid arithmetic inputs")
    return arrays


def encode_gaussian_symbols(symbols, means, stds, steps):
    """Losslessly encode caller-quantized integers; FP64 is only used for CDFs."""
    symbols = np.asarray(symbols).reshape(-1)
    if (not len(symbols) or symbols.dtype.kind not in 'iu'
            or (symbols.dtype.kind == 'u' and symbols.max() > np.iinfo(np.int64).max)):
        raise ValueError("Arithmetic symbols must be nonempty int64-compatible integers")
    symbols = symbols.astype(np.int64, copy=False)
    symbols, means, stds, steps = np.broadcast_arrays(symbols, *_contexts(means, stds, steps))
    minimum, maximum = int(symbols.min()), int(symbols.max())
    if maximum-minimum+1 >= TOTAL:
        raise ValueError("Arithmetic alphabet exceeds precision; reduce chunk size")
    low, high, pending = 0, MASK, 0
    writer = BitWriter()

    def emit(bit):
        nonlocal pending
        writer.write(bit)
        for _ in range(pending):
            writer.write(1-bit)
        pending = 0

    for symbol, mean, std, step in zip(symbols, means, stds, steps):
        a = _boundary(int(symbol), minimum, maximum, mean, std, step)
        b = _boundary(int(symbol)+1, minimum, maximum, mean, std, step)
        span = high-low+1
        high = low+span*b//TOTAL-1
        low = low+span*a//TOTAL
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
    pending += 1
    emit(0 if low < QUARTER else 1)
    return b'GAC1'+struct.pack('<Qqq', len(symbols), minimum, maximum)+writer.finish()


def decode_gaussian_symbols(blob, means, stds, steps):
    """Return exact int64 symbols before any floating-point reconstruction."""
    if blob[:4] != b'GAC1' or len(blob) < 29:
        raise ValueError("Invalid arithmetic stream")
    count, minimum, maximum = struct.unpack('<Qqq', blob[4:28])
    means, stds, steps = _contexts(means, stds, steps)
    if not count or len(means) != count or not 1 <= maximum-minimum+1 < TOTAL:
        raise ValueError("Arithmetic metadata mismatch")
    reader = BitReader(blob[28:])
    low, high, code = 0, MASK, 0
    for _ in range(32):
        code = (code << 1) | reader.read()
    result = np.empty(count, dtype=np.int64)
    for index, (mean, std, step) in enumerate(zip(means, stds, steps)):
        target = ((code-low+1)*TOTAL-1)//(high-low+1)
        left, right = minimum, maximum+1
        while right-left > 1:
            middle = (left+right)//2
            if _boundary(middle, minimum, maximum, mean, std, step) <= target:
                left = middle
            else:
                right = middle
        symbol = left
        a = _boundary(symbol, minimum, maximum, mean, std, step)
        b = _boundary(symbol+1, minimum, maximum, mean, std, step)
        span = high-low+1
        high = low+span*b//TOTAL-1
        low = low+span*a//TOTAL
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
            code = ((code << 1) & MASK) | reader.read()
        result[index] = symbol
    return result


def encode_gaussian(values, means, stds, steps):
    """Convenience wrapper; quantize in the input division dtype, then encode.

    Model export uses encode_gaussian_symbols so it shares the Torch inference
    decision instead of re-quantizing values inside this NumPy wrapper.
    """
    values, steps = np.broadcast_arrays(np.asarray(values).reshape(-1), np.asarray(steps).reshape(-1))
    if not np.isfinite(values).all() or not np.isfinite(steps).all() or (steps <= 0).any():
        raise ValueError("Invalid arithmetic inputs")
    rounded = np.rint(values/steps)
    if not np.isfinite(rounded).all() or (rounded < -(2**63)).any() or (rounded >= 2**63).any():
        raise ValueError("Quantized symbols exceed int64 range")
    return encode_gaussian_symbols(rounded.astype(np.int64), means, stds, steps)


def decode_gaussian(blob, means, stds, steps):
    """Convenience float reconstruction; use the symbol API for integrity checks."""
    symbols = decode_gaussian_symbols(blob, means, stds, steps)
    return (symbols.astype(np.float64)*np.asarray(steps, dtype=np.float64).reshape(-1)).astype(np.float32)
