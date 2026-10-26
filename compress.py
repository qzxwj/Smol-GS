#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json
import torch
from utils.checkpoint import load_checkpoint, save_checkpoint
from smolgs.codec import encode_model, decode_model


def main():
    p = ArgumentParser(description='Compress a trained checkpoint or independently decode a Smol-GS archive')
    sub = p.add_subparsers(dest='command', required=True)
    encode = sub.add_parser('encode')
    encode.add_argument('checkpoint')
    encode.add_argument('output')
    encode.add_argument('--device', default='cuda')
    encode.add_argument('--codec', choices=['cpu', 'cuda'], default='cuda')
    encode.add_argument('--chunk-splats', type=int, default=2048)
    decode = sub.add_parser('decode')
    decode.add_argument('archive')
    decode.add_argument('output')
    decode.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'encode':
        model, _, step, _ = load_checkpoint(args.checkpoint, args.device)
        if step != model.config.iterations:
            raise ValueError('Export requires a completed five-stage checkpoint')
        report = encode_model(model, args.output, args.codec, args.chunk_splats)
    else:
        model, report = decode_model(args.archive, args.device)
        save_checkpoint(args.output, model, None, model.config.iterations,
                        {'decoded_from': str(Path(args.archive).resolve()), 'decode_report': report})
    Path(str(args.output)+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('components')}, indent=2))


if __name__ == '__main__':
    main()
