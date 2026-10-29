"""Standalone ZIP container: occupancy tree, arithmetic attributes, MLPs, binary mixed hash.

No checkpoint, source scene, or SfM data is required for decoding.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import time
import zipfile
import numpy as np
import torch
from utils.arithmetic import encode_gaussian_symbols, decode_gaussian_symbols
from utils.checkpoint import load_torch
from arguments import Config
from utils.formats import ARCHIVE_FORMAT
from scene.gaussian_model import SmolGS
from smolgs.coordinate_grid import encode_coordinates, decode_coordinates


def _hac():
    path = Path(__file__).resolve().parents[1]/'utils/encodings_cuda.py'
    spec = importlib.util.spec_from_file_location('smolgs_hac_codec', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network_files(model):
    # Store only the networks and spatial metadata; never store raw splat features.
    names = ('opacity_mlp.', 'color_mlp.',
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


@torch.no_grad()
def quantized_hash_reference(model):
    """Export reference using exactly the packed binary table signs."""
    reference = SmolGS(Config(**model.config.to_dict()), len(model.xyz), str(model.xyz.device))
    reference.load_state_dict(model.state_dict())
    for name, parameter in reference.hash_encoder.named_parameters():
        parameter.copy_(torch.where(parameter >= 0, 1., -1.))
    reference.eval()
    return reference


def _canonical_model(model, context_device):
    """Create the same backend on the recorded context device."""
    config = Config(**model.config.to_dict())
    if context_device == 'cpu':
        config.hash_backend = 'torch'
    elif config.hash_backend == 'auto':
        # Pin the actual interpolation backend in the archive; never change
        # arithmetic contexts via an automatic fallback on another machine.
        try:
            __import__('_gridencoder')
        except ImportError:
            config.hash_backend = 'torch'
        else:
            config.hash_backend = 'cuda'
    result = SmolGS(config, device=context_device)
    state = {name: value for name, value in model.state_dict().items()
             if name.startswith(('opacity_mlp.', 'color_mlp.',
                                 'entropy_mlp.', 'octree_mlp.', 'hash_encoder.'))}
    result.load_state_dict(state, strict=False)
    result.bounds_min.copy_(model.bounds_min)
    result.bounds_max.copy_(model.bounds_max)
    result.hash_min.copy_(model.hash_min)
    result.hash_max.copy_(model.hash_max)
    result.eval()
    return result


def _pack_cpu(symbols, means, stds, steps):
    return encode_gaussian_symbols(*[a.detach().cpu().numpy().reshape(-1)
                                     for a in (symbols, means, stds, steps)])


def _quantized_symbols(values, steps):
    # Use the model's inference dtype/device, before the CPU coder promotes CDF
    # parameters to FP64. Slicing a stream must never make this decision again.
    rounded = torch.round(values/steps)
    if (not torch.isfinite(rounded).all()
            or (rounded < -(2**63)).any() or (rounded >= 2**63).any()):
        raise ValueError('Nonfinite or out-of-range quantized symbols')
    return rounded.to(torch.int64)


def _attribute_contexts(model, xyz):
    """Only attributes actually quantized by this model get arithmetic streams."""
    mf, sf, qf = model.prior(xyz)
    return {'features': (mf, sf, qf)}


@torch.no_grad()
def _ecvq_geometry(model, order):
    """Assign once, then shrink/remap exactly as RDO shrink_codebook does."""
    if not all(bool(q.initialized) for q in model.quantizers.values()):
        raise ValueError('ECVQ codebooks must be initialized before export')
    result = {}
    for key, values in (('scale', model.log_scales[order].exp()), ('rot', model.rotations[order])):
        quantizer = model.quantizers[key]
        if not torch.isfinite(quantizer.codebook).all() or not torch.isfinite(quantizer.logits).all():
            raise ValueError('Nonfinite ECVQ codebook or logits')
        if key == 'scale':
            # RDO apply_and_save_vq calls remove_negatives before assignment.
            # Apply its exact <0 rule on an export copy; leave training intact.
            quantizer = copy.deepcopy(quantizer)
            quantizer.codebook[quantizer.codebook < 0] = 1e-8
        assigned = quantizer(values, rate_constrain=bool(model.vq_rate_constrained))
        original = assigned['x_index'].cpu().numpy().reshape(-1)
        used, index = np.unique(original, return_inverse=True)
        book = quantizer.codebook[0].detach().cpu().numpy()[used].astype('<f4')
        logits = quantizer.logits[0, torch.as_tensor(used, device=quantizer.logits.device)].cpu()
        result[key] = {'codebook': book, 'indices': index.astype(np.int64)[:, None], 'logits': logits}
    return result


def _npy(value):
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _materialize_geometry(model, geometry):
    """Store lookup results exactly, avoiding exp(log(codeword)) roundoff."""
    for key in ('scale', 'rot'):
        book = torch.as_tensor(geometry[key]['codebook'].copy(), device=model.xyz.device)
        index = torch.as_tensor(geometry[key]['indices'].copy(), device=model.xyz.device, dtype=torch.long)
        quantizer = model.quantizers[key]
        quantizer.codebook = torch.nn.Parameter(book[None], requires_grad=False)
        quantizer.logits = torch.nn.Parameter(torch.zeros(1, len(book), device=book.device), requires_grad=False)
        quantizer.cb_size = len(book)
        quantizer.initialized.fill_(True)
        setattr(model.config, 'vq_scale_cbsize' if key == 'scale' else 'vq_rot_cbsize', len(book))
        setattr(model, key+'_indices', index)
        values = book[index[:, 0]]
        if key == 'scale':
            model.decoded_scales = values
            model.log_scales.copy_(values.log())
        else:
            model.rotations.copy_(values)
    model.vq_rate_constrained.fill_(model.config.iterations >= model.config.rate_constrain_iter)
    model.decoded.fill_(True)


@torch.no_grad()
def export_reference(model, codec='cpu'):
    """Actual export decisions without decoding the resulting bitstream."""
    if model.config.geometry_quantization != 'ecvq':
        return quantized_hash_reference(model)
    _, _, order, canonical_xyz = encode_coordinates(model)
    geometry = _ecvq_geometry(model, order)
    reference = quantized_hash_reference(model)
    reference.replace_topology(order)
    context_device = 'cpu' if codec == 'cpu' else str(model.xyz.device)
    if model.config.appearance_backend == 'tcnn' or model.config.octree_backend == 'tcnn':
        context_device = str(model.xyz.device)
    predictor = _canonical_model(model, context_device)
    xyz = canonical_xyz.to(context_device)
    reference.xyz.copy_(xyz.to(reference.xyz.device))
    _, _, step = predictor.prior(xyz)
    feature = model.features[order].to(context_device)
    reference.features.copy_((torch.round(feature/step)*step).to(reference.xyz.device))
    _materialize_geometry(reference, geometry)
    return reference


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
    if (model.config.appearance_backend == 'tcnn'
            or model.config.octree_backend == 'tcnn'):
        if not model.xyz.is_cuda:
            raise ValueError('tiny-cuda-nn networks/contexts require a CUDA model')
        context_device = str(model.xyz.device)
    if codec == 'cuda' and not model.xyz.is_cuda:
        raise ValueError('HAC CUDA arithmetic coding requires a CUDA model')
    files, binary = _network_files(model)
    predictor = _canonical_model(model, context_device)
    coordinate_files, coordinate_layout, order, canonical_xyz = encode_coordinates(model)
    files.update(coordinate_files)
    # Reconstruct exactly the coordinate tensor that the independent decoder will use.
    xyz = canonical_xyz.to(context_device)
    features = model.features[order].to(context_device)
    attributes = {'features': features}
    geometry_meta = {}
    if model.config.geometry_quantization == 'ecvq':
        from smolgs.vq.coding import cdf_from_logits, encode_indices, decode_indices
        for key, data in _ecvq_geometry(model, order).items():
            cdf = cdf_from_logits(data['logits'])
            stream = encode_indices(data['indices'], cdf)
            if not np.array_equal(decode_indices(stream, cdf, len(xyz)), data['indices']):
                raise RuntimeError('ECVQ index roundtrip mismatch')
            prefix = 'geometry/'+key
            files[prefix+'_codebook.npy'] = _npy(data['codebook'])
            files[prefix+'_cdf.npy'] = _npy(cdf)
            files[prefix+'_indices.rans'] = stream
            geometry_meta[key] = {'codes': len(data['codebook']), 'codebook': prefix+'_codebook.npy',
                                 'cdf': prefix+'_cdf.npy', 'indices': prefix+'_indices.rans'}
    else:
        # Step-one geometry has no lossy quantization or learned rate model.
        # Save raw FP32 log scales/quaternions in exactly the octree order.
        for name in ('log_scales', 'rotations'):
            value = getattr(model, name)[order].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite explicit geometry: {name}')
            if name == 'log_scales':
                physical = torch.from_numpy(value).exp()
                if not torch.isfinite(physical).all() or not (physical > 0).all():
                    raise ValueError('Explicit geometry requires finite positive scales')
            buffer = io.BytesIO()
            np.save(buffer, value.astype('<f4'), allow_pickle=False)
            files[f'geometry/{name}.npy'] = buffer.getvalue()
    # Use the same complete-scene batch as checkpoint inference. Re-evaluating
    # the entropy MLP on adaptive stream chunks changes floating-point GEMM
    # results slightly; near half-integers this can change quantized symbols.
    # Chunking below controls arithmetic-stream memory, not the learned context.
    contexts = _attribute_contexts(predictor, xyz)
    chunks = []
    hac = _hac() if codec == 'cuda' else None
    with tempfile.TemporaryDirectory(prefix='smolgs-codec-') as temporary:
        def pack(start, end, symbols):
            # HAC uses signed int16 symbols and a dense [N,alphabet] CDF.
            spans = {name: int(value.max())-int(value.min())+1 for name, value in symbols.items()}
            limit = 32767 if codec == 'cuda' else (1 << 20)-1
            if max(spans.values()) > limit or (codec == 'cuda' and max(
                    symbols[name].numel()*span for name, span in spans.items()) > 4_000_000):
                if end-start > 1:
                    middle = (start+end)//2
                    offset = middle-start
                    pack(start, middle, {name: value[:offset] for name, value in symbols.items()})
                    pack(middle, end, {name: value[offset:] for name, value in symbols.items()})
                    return
                # A single splat can still span too many quantized symbols across
                # its channels. Split that attribute into smaller HAC streams;
                # retain the exact quantization and count every stream in size.
            index = len(chunks)
            chunk = {'start': start, 'end': end}
            for label, source in attributes.items():
                values = source[start:end]
                means, stds, steps = (value[start:end] for value in contexts[label])
                steps = steps.expand_as(values).contiguous()
                integers = symbols[label]
                name = f'{label}/{index:06d}.bin'
                flat = [value.reshape(-1).contiguous() for value in (integers, values, means, stds, steps)]
                parts = []
                def pack_attribute(first, last):
                    k, v, m, sd, q = [value[first:last].contiguous() for value in flat]
                    span = int(k.max())-int(k.min())+1
                    if span > limit or (codec == 'cuda' and (last-first)*span > 4_000_000):
                        if last-first == 1:
                            raise ValueError('Nonfinite or unsupported scalar quantization range')
                        middle = (first+last)//2
                        pack_attribute(first, middle)
                        pack_attribute(middle, last)
                        return
                    filename = name if first == 0 and last == values.numel() else f'{label}/{index:06d}_{len(parts):03d}.bin'
                    if codec == 'cpu':
                        files[filename] = _pack_cpu(k, m, sd, q)
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
            pack(start, end, {name: _quantized_symbols(value[start:end], contexts[name][2][start:end])
                              for name, value in attributes.items()})
            if start//chunk_splats % 20 == 0 or end == len(xyz):
                print(f'[Encode] {end:,}/{len(xyz):,} splats ({100*end/len(xyz):.1f}%)', flush=True)
    segmented = sum(isinstance(chunk[label], list) for chunk in chunks for label in attributes)
    metadata = {'format': ARCHIVE_FORMAT, 'geometry_mode': model.config.geometry_mode,
                'geometry_quantization': model.config.geometry_quantization,
                'coordinate_layout': coordinate_layout,
                'geometry_storage': 'rdo-rans16-v1' if geometry_meta else
                                    'lossless-fp32',
                'geometry': geometry_meta,
                'hash_model': 'hac-binary-mixed-detached-v1',
                'context_layout': 'full-scene-v1',
                'implementation': 'independent-paper-reproduction',
                'codec': codec, 'context_device': 'cuda' if str(context_device).startswith('cuda') else 'cpu',
                'config': predictor.config.to_dict(), 'count': len(xyz), 'binary_hash': binary,
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
              'codec': codec, 'hash_model': 'hac-binary-mixed-detached-v1', 'segmented_attributes': segmented, 'components_raw_bytes': {name: len(data) for name, data in files.items()}}
    report.update(geometry_mode=metadata['geometry_mode'], geometry_storage=metadata['geometry_storage'])
    return report


@torch.no_grad()
def decode_model(path, device='cpu'):
    started = time.perf_counter()
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read('metadata.json'))
        if meta.get('format') != ARCHIVE_FORMAT:
            raise ValueError(f'Unsupported archive format; expected {ARCHIVE_FORMAT}')
        if meta.get('hash_model') != 'hac-binary-mixed-detached-v1' or meta.get('context_layout') != 'full-scene-v1':
            raise ValueError('Unsupported hash model or context layout')
        files = {name: archive.read(name) for name in meta['files_sha256']}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != meta['files_sha256'][name]:
            raise ValueError(f'Archive checksum mismatch: {name}')
    config = Config.from_saved(meta['config'])
    is_ecvq = config.geometry_quantization == 'ecvq'
    if meta.get('geometry_quantization') != config.geometry_quantization:
        raise ValueError('Archive geometry quantization/config mismatch')
    if meta.get('geometry_mode') != config.geometry_mode:
        raise ValueError('Archive geometry mode/config mismatch')
    expected_storage = ('rdo-rans16-v1' if is_ecvq else
                        'lossless-fp32')
    if meta.get('geometry_storage') != expected_storage:
        raise ValueError('Unsupported geometry storage')
    geometry = {}
    if is_ecvq:
        from smolgs.vq.coding import decode_indices, validate_cdf
        if set(meta.get('geometry', {})) != {'scale', 'rot'} or meta['count'] < 1:
            raise ValueError('Invalid ECVQ geometry metadata')
        for key, width in (('scale', 3), ('rot', 4)):
            description = meta['geometry'][key]
            book = np.load(io.BytesIO(files[description['codebook']]), allow_pickle=False)
            codes = description['codes']
            if (not isinstance(codes, int) or not 0 < codes < 65536 or book.shape != (codes, width)
                    or book.dtype != np.dtype('<f4') or not np.isfinite(book).all()
                    or (key == 'scale' and not (book >= 0).all())):
                raise ValueError('Invalid ECVQ codebook')
            cdf = validate_cdf(np.load(io.BytesIO(files[description['cdf']]), allow_pickle=False), codes)
            index = decode_indices(files[description['indices']], cdf, meta['count'])
            geometry[key] = {'codebook': book, 'indices': index}
            setattr(config, 'vq_scale_cbsize' if key == 'scale' else 'vq_rot_cbsize', codes)
    context_device = 'cpu' if meta['context_device'] == 'cpu' else device
    if meta['context_device'] == 'cuda' and not str(device).startswith('cuda'):
        raise ValueError('This archive requires --device cuda for its native networks/contexts')
    if meta['codec'] == 'cuda' and not str(device).startswith('cuda'):
        raise ValueError('CUDA arithmetic coding requires --device cuda')
    if context_device == 'cpu':
        config.hash_backend = 'torch'
    model = SmolGS(config, meta['count'], context_device)
    weights = load_torch(io.BytesIO(files['networks.pt']))
    expected_tables = {name: value for name, value in model.state_dict().items()
                       if name.startswith('hash_encoder.') and name.endswith('.table')}
    if set(meta['binary_hash']) != set(expected_tables):
        raise ValueError('Binary hash table set mismatch')
    for name, info in meta['binary_hash'].items():
        if info['shape'] != list(expected_tables[name].shape) or info['count'] != expected_tables[name].numel():
            raise ValueError('Binary hash table shape mismatch')
        data = files[info['file']]
        if len(data) != (info['count']+7)//8:
            raise ValueError('Binary hash payload length mismatch')
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little')[:info['count']]
        weights[name] = torch.from_numpy((bits.astype(np.float32)*2-1).reshape(info['shape']))
    model.load_state_dict(weights, strict=False)
    for name in ('bounds_min', 'bounds_max', 'hash_min', 'hash_max'):
        getattr(model, name).copy_(torch.tensor(meta[name], device=context_device))
    decode_coordinates(model, files, meta['coordinate_layout'], meta['count'])
    if is_ecvq:
        _materialize_geometry(model, geometry)
    else:
        for name, width in (('log_scales', 3), ('rotations', 4)):
            value = np.load(io.BytesIO(files[f'geometry/{name}.npy']), allow_pickle=False)
            if value.shape != (meta['count'], width) or value.dtype != np.dtype('<f4') or not np.isfinite(value).all():
                raise ValueError(f'Invalid explicit geometry payload: {name}')
            tensor = torch.as_tensor(value, device=context_device)
            if name == 'log_scales' and (not torch.isfinite(tensor.exp()).all() or not (tensor.exp() > 0).all()):
                raise ValueError('Explicit geometry requires finite positive scales')
            getattr(model, name).copy_(tensor)
    model.eval()
    contexts = _attribute_contexts(model, model.xyz)
    hac = _hac() if meta['codec'] == 'cuda' else None
    expected_start = 0
    with tempfile.TemporaryDirectory(prefix='smolgs-decode-') as temporary:
        for chunk in meta['chunks']:
            start, end = chunk['start'], chunk['end']
            if start != expected_start or not start < end <= meta['count']:
                raise ValueError('Noncontiguous attribute chunks')
            expected_start = end
            for label, context in contexts.items():
                means, stds, steps = (value[start:end] for value in context)
                steps = steps.expand_as(means).contiguous()
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
                        symbols = decode_gaussian_symbols(data, *[a.cpu().numpy() for a in (m, sd, q)])
                        decoded = torch.as_tensor(symbols, device=q.device, dtype=q.dtype)*q
                    else:
                        filename = Path(temporary)/'stream.b'
                        filename.write_bytes(data)
                        decoded = hac.decoder_gaussian(m, sd, q, str(filename))
                    values.reshape(-1)[first:last].copy_(decoded)
                if cursor != means.numel():
                    raise ValueError('Missing attribute segments')
                model.features[start:end].copy_(values)
    if expected_start != meta['count']:
        raise ValueError('Missing attribute chunks')
    model.coordinate_frozen.fill_(True)
    model.decoded.fill_(True)
    model.xyz.requires_grad_(False)
    model.to(device)
    model.config.hash_backend = meta['config']['hash_backend']
    return model, {'decode_seconds': time.perf_counter()-started, 'splats': len(model.xyz),
                   'size_MB': Path(path).stat().st_size/1e6, 'codec': meta['codec'],
                   'geometry_mode': config.geometry_mode,
                   'geometry_storage': meta['geometry_storage']}
