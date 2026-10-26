"""Standalone ZIP container: occupancy tree, arithmetic attributes, MLPs, binary hash.

No checkpoint, source scene, or SfM data is required for decoding.
"""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import tempfile
import time
import zipfile
import numpy as np
import torch
from utils.arithmetic import encode_gaussian, decode_gaussian
from utils.checkpoint import load_torch
from arguments import Config
from scene.gaussian_model import SmolGS
from utils.octree import encode_octree, decode_octree


def _hac():
    path = Path(__file__).resolve().parents[1]/'utils/encodings_cuda.py'
    spec = importlib.util.spec_from_file_location('smolgs_hac_codec', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network_files(model):
    # Store only the networks and spatial metadata; never store raw splat features.
    names = ('opacity_mlp.', 'color_mlp.', 'rotation_mlp.', 'scale_mlp.',
             'entropy_mlp.', 'octree_mlp.', 'hash_encoder.')
    state = {}
    binary = {}
    files = {}
    for name, value in model.state_dict().items():
        if not name.startswith(names):
            continue
        if name.startswith('hash_encoder.') and name.endswith('.table'):
            bits = (value.detach().cpu().numpy().reshape(-1) >= 0).astype(np.uint8)
            filename = 'hash/'+name+'.bin'
            files[filename] = np.packbits(bits, bitorder='little').tobytes()
            binary[name] = {'file': filename, 'shape': list(value.shape), 'count': len(bits)}
        else:
            state[name] = value.detach().cpu()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    files['networks.pt'] = buffer.getvalue()
    return files, binary


def _canonical_model(model, context_device):
    """Binary hash weights are identical in the training and stored networks."""
    config = Config(**model.config.to_dict())
    # CPU context is deterministic across render devices; CUDA uses HAC's native grid.
    config.hash_backend = 'torch' if context_device == 'cpu' else 'cuda'
    result = SmolGS(config, device=context_device)
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith(('opacity_mlp.', 'color_mlp.', 'rotation_mlp.', 'scale_mlp.',
                                 'entropy_mlp.', 'octree_mlp.', 'hash_encoder.'))}
    result.load_state_dict(state, strict=False)
    result.bounds_min.copy_(model.bounds_min)
    result.bounds_max.copy_(model.bounds_max)
    result.hash_min.copy_(model.hash_min)
    result.hash_max.copy_(model.hash_max)
    result.eval()
    return result


def _pack_cpu(values, means, stds, steps):
    return encode_gaussian(*[a.detach().cpu().numpy().reshape(-1) for a in (values, means, stds, steps)])


@torch.no_grad()
def encode_model(model, output, codec='cpu', chunk_splats=2048):
    if not bool(model.coordinate_frozen):
        raise ValueError('Coordinate compression stage is required before export')
    if bool(model.decoded):
        raise ValueError('Encode the training checkpoint, not an already decoded archive')
    if codec not in ('cpu', 'cuda') or chunk_splats < 1:
        raise ValueError('Invalid codec or chunk size')
    started = time.perf_counter()
    print(f'[Encode] {len(model.xyz):,} splats, codec={codec}, output={output}', flush=True)
    context_device = 'cpu' if codec == 'cpu' else str(model.xyz.device)
    if codec == 'cuda' and not model.xyz.is_cuda:
        raise ValueError('HAC CUDA arithmetic coding requires a CUDA model')
    predictor = _canonical_model(model, context_device)
    files, binary = _network_files(model)
    octree, order = encode_octree(model.cells.cpu().numpy(), model.config.octree_depth)
    files['coordinates.oct'] = octree
    order = torch.as_tensor(order.astype(np.int64), device=model.xyz.device)
    cells = model.cells[order].to(context_device)
    # Reconstruct exactly the coordinate tensor that the independent decoder will use.
    xyz = predictor.bounds_min+(cells.to(torch.float32)+0.5)*(
        predictor.bounds_max-predictor.bounds_min).clamp_min(1e-6)/(1 << model.config.octree_depth)
    features = model.features[order].to(context_device)
    scales = model.log_scales[order].exp().to(context_device)
    # Use the same complete-scene batch as checkpoint inference. Re-evaluating
    # the entropy MLP on adaptive stream chunks changes floating-point GEMM
    # results slightly; near half-integers this can change quantized symbols.
    # Chunking below controls arithmetic-stream memory, not the learned context.
    contexts = predictor.prior(xyz)
    chunks = []
    hac = _hac() if codec == 'cuda' else None
    with tempfile.TemporaryDirectory(prefix='smolgs-codec-') as temporary:
        def pack(start, end):
            mf, sf, ms, ss, qf, qs = (value[start:end] for value in contexts)
            f, s = features[start:end], scales[start:end]
            qf, qs = qf.expand_as(f).contiguous(), qs.expand_as(s).contiguous()
            # HAC uses signed int16 symbols and a dense [N,alphabet] CDF.
            spans = [int((v/q).round().max()-(v/q).round().min())+1 for v, q in ((f, qf), (s, qs))]
            limit = 32767 if codec == 'cuda' else (1 << 20)-1
            if max(spans) > limit or (codec == 'cuda' and max((end-start)*model.config.feature_dim*spans[0],
                                                                           (end-start)*3*spans[1]) > 4_000_000):
                if end-start > 1:
                    middle = (start+end)//2
                    pack(start, middle)
                    pack(middle, end)
                    return
                # A single splat can still span too many quantized symbols across
                # its channels. Split that attribute into smaller HAC streams;
                # retain the exact quantization and count every stream in size.
            index = len(chunks)
            chunk = {'start': start, 'end': end}
            for label, values, means, stds, steps in [('features', f, mf, sf, qf), ('scales', s, ms, ss, qs)]:
                name = f'{label}/{index:06d}.bin'
                symbols = torch.round(values/steps).to(torch.int64).cpu().numpy().astype('<i8')
                chunk[label+'_sha256'] = hashlib.sha256(symbols.tobytes()).hexdigest()
                flat = [value.reshape(-1).contiguous() for value in (values, means, stds, steps)]
                parts = []
                def pack_attribute(first, last):
                    v, m, sd, q = [value[first:last].contiguous() for value in flat]
                    integers = torch.round(v/q)
                    span = int(integers.max()-integers.min())+1
                    if span > limit or (codec == 'cuda' and (last-first)*span > 4_000_000):
                        if last-first == 1:
                            raise ValueError('Nonfinite or unsupported scalar quantization range')
                        middle = (first+last)//2
                        pack_attribute(first, middle)
                        pack_attribute(middle, last)
                        return
                    filename = name if first == 0 and last == values.numel() else f'{label}/{index:06d}_{len(parts):03d}.bin'
                    if codec == 'cpu':
                        files[filename] = _pack_cpu(v, m, sd, q)
                    else:
                        path = str(Path(temporary)/'stream.b')
                        hac.encoder_gaussian(v, m, sd, q, path)
                        files[filename] = Path(path).read_bytes()
                    parts.append({'start': first, 'end': last, 'file': filename})
                pack_attribute(0, values.numel())
                chunk[label] = parts[0]['file'] if len(parts) == 1 else parts
            chunks.append(chunk)
        for start in range(0, len(xyz), chunk_splats):
            end = min(start+chunk_splats, len(xyz))
            pack(start, end)
            if start//chunk_splats % 20 == 0 or end == len(xyz):
                print(f'[Encode] {end:,}/{len(xyz):,} splats ({100*end/len(xyz):.1f}%)', flush=True)
    segmented = sum(isinstance(chunk[label], list) for chunk in chunks for label in ('features', 'scales'))
    metadata = {'format': 'smolgs-archive-v3', 'context_layout': 'full-scene-v1',
                'implementation': 'independent-paper-reproduction',
                'codec': codec, 'context_device': 'cpu' if codec == 'cpu' else 'cuda',
                'config': model.config.to_dict(), 'count': len(xyz), 'binary_hash': binary,
                'bounds_min': model.bounds_min.tolist(), 'bounds_max': model.bounds_max.tolist(),
                'hash_min': model.hash_min.tolist(), 'hash_max': model.hash_max.tolist(),
                'chunks': chunks, 'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files['metadata.json'] = (json.dumps(metadata, indent=2)+'\n').encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    temporary.replace(output)
    report = {'archive': str(output.resolve()), 'splats': len(xyz),
              'bytes': output.stat().st_size, 'size_MB': output.stat().st_size/1e6,
              'size_MiB': output.stat().st_size/2**20, 'encode_seconds': time.perf_counter()-started,
              'codec': codec, 'segmented_attributes': segmented, 'components_raw_bytes': {name: len(data) for name, data in files.items()}}
    return report


@torch.no_grad()
def decode_model(path, device='cpu'):
    started = time.perf_counter()
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read('metadata.json'))
        if meta.get('format') not in ('smolgs-archive-v1', 'smolgs-archive-v2', 'smolgs-archive-v3'):
            raise ValueError('Unsupported archive format')
        if meta['format'] == 'smolgs-archive-v3' and meta.get('context_layout') != 'full-scene-v1':
            raise ValueError('Unsupported archive context layout')
        files = {name: archive.read(name) for name in meta['files_sha256']}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != meta['files_sha256'][name]:
            raise ValueError(f'Archive checksum mismatch: {name}')
    config = Config(**meta['config'])
    context_device = 'cpu' if meta['codec'] == 'cpu' else device
    if meta['codec'] == 'cuda' and not str(device).startswith('cuda'):
        raise ValueError('This archive uses HAC CUDA arithmetic coding; decode with --device cuda')
    config.hash_backend = 'torch' if context_device == 'cpu' else 'cuda'
    model = SmolGS(config, meta['count'], context_device)
    weights = load_torch(io.BytesIO(files['networks.pt']))
    for name, info in meta['binary_hash'].items():
        bits = np.unpackbits(np.frombuffer(files[info['file']], dtype=np.uint8), bitorder='little')[:info['count']]
        weights[name] = torch.from_numpy((bits.astype(np.float32)*2-1).reshape(info['shape']))
    model.load_state_dict(weights, strict=False)
    for name in ('bounds_min', 'bounds_max', 'hash_min', 'hash_max'):
        getattr(model, name).copy_(torch.tensor(meta[name], device=context_device))
    cells, depth = decode_octree(files['coordinates.oct'])
    if depth != config.octree_depth or len(cells) != meta['count']:
        raise ValueError('Archive coordinate metadata mismatch')
    model.cells = torch.as_tensor(cells, device=context_device)
    model.xyz.copy_(model.bounds_min+(model.cells.to(torch.float32)+0.5)*(
        model.bounds_max-model.bounds_min).clamp_min(1e-6)/(1 << depth))
    model.eval()
    # v1/v2 contexts were computed separately for each adaptive chunk. Preserve
    # that convention when decoding existing archives, including their CDFs.
    contexts = model.prior(model.xyz) if meta['format'] == 'smolgs-archive-v3' else None
    hac = _hac() if meta['codec'] == 'cuda' else None
    expected_start = 0
    with tempfile.TemporaryDirectory(prefix='smolgs-decode-') as temporary:
        for chunk in meta['chunks']:
            start, end = chunk['start'], chunk['end']
            if start != expected_start or not start < end <= meta['count']:
                raise ValueError('Noncontiguous attribute chunks')
            expected_start = end
            mf, sf, ms, ss, qf, qs = ((value[start:end] for value in contexts) if contexts is not None
                                     else model.prior(model.xyz[start:end]))
            qf, qs = qf.expand_as(mf).contiguous(), qs.expand_as(ms).contiguous()
            for label, means, stds, steps in [('features', mf, sf, qf), ('scales', ms, ss, qs)]:
                description = chunk[label]
                parts = ([{'start': 0, 'end': means.numel(), 'file': description}]
                         if isinstance(description, str) else description)
                values = torch.empty_like(means)
                flattened = [a.reshape(-1) for a in (means, stds, steps)]
                cursor = 0
                for part in parts:
                    first, last = part['start'], part['end']
                    if first != cursor or not first < last <= means.numel():
                        raise ValueError('Noncontiguous attribute segments')
                    cursor = last
                    m, sd, q = [a[first:last].contiguous() for a in flattened]
                    data = files[part['file']]
                    if meta['codec'] == 'cpu':
                        decoded = decode_gaussian(data, *[a.cpu().numpy() for a in (m, sd, q)])
                        decoded = torch.as_tensor(decoded, device=context_device)
                    else:
                        filename = Path(temporary)/'stream.b'
                        filename.write_bytes(data)
                        decoded = hac.decoder_gaussian(m, sd, q, str(filename))
                    values.reshape(-1)[first:last].copy_(decoded)
                if cursor != means.numel():
                    raise ValueError('Missing attribute segments')
                symbols = torch.round(values/steps).to(torch.int64).cpu().numpy().astype('<i8')
                if hashlib.sha256(symbols.tobytes()).hexdigest() != chunk[label+'_sha256']:
                    raise ValueError(f'Arithmetic roundtrip failed for {label}, chunk {start}:{end}')
                if label == 'features':
                    model.features[start:end].copy_(values)
                else:
                    model.log_scales[start:end].copy_(values.clamp_min(1e-7).log())
    if expected_start != meta['count']:
        raise ValueError('Missing attribute chunks')
    model.coordinate_frozen.fill_(True)
    model.decoded.fill_(True)
    model.xyz.requires_grad_(False)
    model.to(device)
    model.config.hash_backend = meta['config']['hash_backend']
    return model, {'decode_seconds': time.perf_counter()-started, 'splats': len(model.xyz),
                   'size_MB': Path(path).stat().st_size/1e6, 'codec': meta['codec']}
