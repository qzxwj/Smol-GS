"""RDO-Gaussian's 16-bit unconditional rANS, with serialized integer CDFs."""
import numpy as np
import torch


def native():
    try:
        from smolgs_rdo_entropy import rans, _CXX
    except ImportError as error:
        raise RuntimeError('Install RDO entropy extensions: python -m pip install --no-build-isolation --no-deps ./submodules/rdo-entropy') from error
    return rans, _CXX


def cdf_from_logits(logits):
    _, ops = native()
    # Exactly the reference _build_tables: softmax, overflow symbol, precision 16.
    p = torch.softmax(logits.detach().float().cpu().reshape(-1), dim=0)
    if not len(p) or len(p) >= 65536 or not torch.isfinite(p).all():
        raise ValueError('Invalid rANS codebook probabilities')
    p = torch.cat([p, (1-p.sum(keepdim=True, dim=0)).clamp_min(0)])
    return np.asarray(ops.pmf_to_quantized_cdf(p.tolist(), 16), dtype='<i4')


def validate_cdf(cdf, codes):
    cdf = np.asarray(cdf)
    if (cdf.dtype.kind not in 'iu' or cdf.shape != (codes+2,) or cdf[0] != 0
            or cdf[-1] != 65536 or not (np.diff(cdf.astype(np.int64)) > 0).all()):
        raise ValueError('Invalid rANS integer CDF')
    return cdf.astype('<i4')


def encode_indices(indices, cdf):
    rans, _ = native()
    indices = np.asarray(indices)
    codes = len(cdf)-2
    cdf = validate_cdf(cdf, codes)
    if indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= codes).any():
        raise ValueError('Out-of-range geometry indices')
    return rans.RansEncoder().encode_with_indexes(indices.reshape(-1).tolist(),
                                                  [0]*indices.size, [cdf.tolist()], [len(cdf)], [0])


def decode_indices(stream, cdf, count):
    rans, _ = native()
    cdf = validate_cdf(cdf, len(cdf)-2)
    if not isinstance(count, int) or count < 0 or len(stream) < 8 or len(stream) % 4:
        raise ValueError('Invalid geometry rANS stream metadata')
    values = np.asarray(rans.RansDecoder().decode_with_indexes(stream, [0]*count,
                        [cdf.tolist()], [len(cdf)], [0]), dtype=np.int64).reshape(count, 1)
    if (values < 0).any() or (values >= len(cdf)-2).any():
        raise ValueError('Decoded geometry index out of range')
    return values
