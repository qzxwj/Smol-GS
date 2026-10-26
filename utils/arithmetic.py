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


def encode_gaussian(values, means, stds, steps):
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (values, means, stds, steps)]
    values, means, stds, steps = np.broadcast_arrays(*arrays)
    if not len(values) or not all(np.isfinite(a).all() for a in arrays) or (steps <= 0).any():
        raise ValueError("Invalid arithmetic inputs")
    symbols = np.rint(values/steps).astype(np.int64)
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


def decode_gaussian(blob, means, stds, steps):
    if blob[:4] != b'GAC1' or len(blob) < 29:
        raise ValueError("Invalid arithmetic stream")
    count, minimum, maximum = struct.unpack('<Qqq', blob[4:28])
    means, stds, steps = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64).reshape(-1)
                                               for a in (means, stds, steps)])
    if len(means) != count or maximum-minimum+1 >= TOTAL:
        raise ValueError("Arithmetic metadata mismatch")
    reader = BitReader(blob[28:])
    low, high, code = 0, MASK, 0
    for _ in range(32):
        code = (code << 1) | reader.read()
    result = np.empty(count, dtype=np.float32)
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
        result[index] = symbol*step
    return result
