"""Breadth-first occupancy bytes followed by canonical Huffman coding."""
import heapq
import struct
import numpy as np


def _codes(lengths):
    code, previous = 0, 0
    result = {}
    for length, symbol in sorted((int(n), s) for s, n in enumerate(lengths) if n):
        code <<= length-previous
        result[symbol] = (code, length)
        code += 1
        previous = length
    return result


def huffman_encode(data):
    values = np.frombuffer(data, dtype=np.uint8)
    frequencies = np.bincount(values, minlength=256)
    heap = [(int(n), s, s) for s, n in enumerate(frequencies) if n]
    heapq.heapify(heap)
    serial = 256
    while len(heap) > 1:
        a, _, left = heapq.heappop(heap)
        b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (a+b, serial, (left, right)))
        serial += 1
    lengths = [0]*256
    if heap:
        stack = [(heap[0][2], 0)]
        while stack:
            node, depth = stack.pop()
            if isinstance(node, int):
                lengths[node] = max(1, depth)
            else:
                stack.extend([(node[0], depth+1), (node[1], depth+1)])
    codes = _codes(lengths)
    out = bytearray()
    accumulator = bits = 0
    for symbol in values:
        code, size = codes[int(symbol)]
        accumulator = (accumulator << size) | code
        bits += size
        while bits >= 8:
            bits -= 8
            out.append((accumulator >> bits) & 255)
        accumulator &= (1 << bits)-1
    if bits:
        out.append((accumulator << (8-bits)) & 255)
    return struct.pack('<Q', len(values))+bytes(lengths)+out


def huffman_decode(blob):
    if len(blob) < 264:
        raise ValueError("Truncated Huffman header")
    count, = struct.unpack('<Q', blob[:8])
    if count == 0:
        return b''
    codes = _codes(blob[8:264])
    lookup = {(length, code): symbol for symbol, (code, length) in codes.items()}
    max_length = max(blob[8:264])
    out = bytearray()
    code = length = 0
    for byte in blob[264:]:
        for bit in range(7, -1, -1):
            code = (code << 1) | ((byte >> bit) & 1)
            length += 1
            symbol = lookup.get((length, code))
            if symbol is not None:
                out.append(symbol)
                code = length = 0
                if len(out) == count:
                    return bytes(out)
            elif length > max_length:
                raise ValueError("Invalid Huffman payload")
    raise ValueError("Truncated Huffman payload")


def numpy_morton(cells, depth):
    cells = np.asarray(cells, dtype=np.uint64)
    codes = np.zeros(len(cells), dtype=np.uint64)
    for bit in range(depth-1, -1, -1):
        xyz = (cells >> bit) & 1
        codes = (codes << 3) | (xyz[:, 0]*4+xyz[:, 1]*2+xyz[:, 2])
    return codes


def encode_octree(cells, depth):
    """Returns encoded bytes and Morton order (features must use the same order)."""
    cells = np.asarray(cells, dtype=np.int64)
    if not 1 <= depth <= 20 or cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("Invalid octree depth or cells shape")
    if not len(cells) or (cells < 0).any() or (cells >= 1 << depth).any():
        raise ValueError("Octree requires nonempty in-range integer cells")
    codes = numpy_morton(cells, depth)
    order = np.argsort(codes, kind='stable')
    codes = codes[order]
    if np.any(codes[1:] == codes[:-1]):
        raise ValueError("Duplicate leaves must be pruned before octree encoding")
    levels = []
    for level in range(1, depth+1):
        prefixes = np.unique(codes >> (3*(depth-level)))
        parents, children = prefixes >> 3, (prefixes & 7).astype(np.uint8)
        _, inverse = np.unique(parents, return_inverse=True)
        occupancy = np.zeros(int(inverse.max())+1, dtype=np.uint8)
        np.bitwise_or.at(occupancy, inverse, (1 << children).astype(np.uint8))
        levels.append(occupancy.tobytes())
    return b'OCT1'+struct.pack('<BQ', depth, len(codes))+huffman_encode(b''.join(levels)), order


def decode_octree(blob):
    if blob[:4] != b'OCT1' or len(blob) < 13:
        raise ValueError("Invalid octree header")
    depth, count = struct.unpack('<BQ', blob[4:13])
    if not 1 <= depth <= 20:
        raise ValueError("Invalid octree depth")
    occupancy = np.frombuffer(huffman_decode(blob[13:]), dtype=np.uint8)
    prefixes = np.zeros(1, dtype=np.uint64)
    offset = 0
    for _ in range(depth):
        nodes = len(prefixes)
        if offset+nodes > len(occupancy):
            raise ValueError("Truncated occupancy tree")
        codes = occupancy[offset:offset+nodes]
        if not codes.all():
            raise ValueError("Empty internal node in occupancy tree")
        parent, child = np.nonzero((codes[:, None] >> np.arange(8)) & 1)
        prefixes = (prefixes[parent] << 3) | child.astype(np.uint64)
        offset += nodes
    if len(prefixes) != count or offset != len(occupancy):
        raise ValueError("Octree leaf count/payload mismatch")
    cells = np.zeros((count, 3), dtype=np.int64)
    for bit in range(depth):
        child = (prefixes >> (3*bit)) & 7
        cells[:, 0] |= ((child >> 2) & 1).astype(np.int64) << bit
        cells[:, 1] |= ((child >> 1) & 1).astype(np.int64) << bit
        cells[:, 2] |= (child & 1).astype(np.int64) << bit
    return cells, depth
