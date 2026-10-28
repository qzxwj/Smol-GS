"""HAC binary mixed grid, detached coordinates, and standalone bit-packed export."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from arguments import Config, config_from_args
from gaussian_renderer import render
from scene.dataset_readers import Camera
from scene.gaussian_model import SmolGS
from smolgs.codec import encode_model, decode_model, quantized_hash_reference, _canonical_model
from train import parser
from utils.checkpoint import load_checkpoint, save_checkpoint, load_torch
from utils.hashgrid import BinarySTE, MixedHashEncoder
from utils.loss_utils import loss_terms


def config(backend='torch'):
    return Config(mlp_width=16, octree_mlp_width=8,
                  resolutions_3d=(4, 6), resolutions_2d=(4,), hash_features=2,
                  hash_log2_3d=5, hash_log2_2d=5,
                  iterations=25, warmup_until=2, densify_until=8, compact_until=12,
                  feature_until=18, lr_max_steps=18, adc_interval=2, hash_backend=backend)


class BinaryHashTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(34)

    def model(self, backend='torch', device='cpu'):
        model=SmolGS(config(backend),device=device)
        points=torch.randn(12,3,device=device)*0.12
        points[:,2]+=2
        model.initialize(points)
        model.set_scene_extent(torch.tensor([[-2.,0.,0.],[2.,0.,0.]],device=device))
        return model

    def test_default_layout_configs_and_tcnn_hash_rejection(self):
        args=parser().parse_args(['-s','/tmp/data','-m','/tmp/model'])
        current=config_from_args(args)
        self.assertEqual(current.hash_backend,'auto')
        self.assertTrue(current.hash_detach_coordinates)
        grid=MixedHashEncoder(current)
        self.assertEqual(grid.output_dim,96)
        self.assertEqual((grid.xyz.output_dim,grid.xy.output_dim),(48,16))
        self.assertEqual(current.resolutions_3d,(18,24,33,44,59,80,108,148,201,275,376,514))
        self.assertEqual(current.resolutions_2d,(130,258,514,1026))
        for path in (Path(__file__).resolve().parents[1]/'configs').glob('*.json'):
            loaded=Config.load(path)
            self.assertNotEqual(loaded.hash_backend,'tcnn')
            self.assertTrue(loaded.hash_detach_coordinates)
        with self.assertRaisesRegex(ValueError,'Binary hash_backend'): Config(hash_backend='tcnn')
        with self.assertRaisesRegex(ValueError,'requires hash_detach'): Config(hash_detach_coordinates=False)

    def test_binary_ste_zero_sign_and_clipped_gradient(self):
        x=torch.tensor([-2.,-1.,-0.1,0.,0.1,1.,2.],requires_grad=True)
        y=BinarySTE.apply(x)
        torch.testing.assert_close(y,torch.tensor([-1.,-1.,-1.,1.,1.,1.,1.]),atol=0,rtol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([0.,1.,1.,1.,1.,1.,0.]),atol=0,rtol=0)

    def test_planes_boundary_normalization_and_detached_input(self):
        grid=MixedHashEncoder(config())
        with torch.no_grad():
            for name,table in grid.named_parameters():
                table.fill_(-0.25 if name.startswith(('xy.','yz.')) else 0.)
        points=torch.tensor([[0.,0.,0.],[1.,1.,1.],[0.01,0.43,0.99]],requires_grad=True)
        values=grid(points)
        expected=torch.tensor([1.,1.,1.,1.,-1.,-1.,1.,1.,-1.,-1.]).expand(3,-1)
        torch.testing.assert_close(values,expected,atol=1e-6,rtol=0)
        # Out-of-domain 3D coordinates produce zero 3D context.
        torch.testing.assert_close(grid.xyz(torch.tensor([[-0.1,0.5,0.5],[1.1,0.5,0.5]])),torch.zeros(2,4))
        values.sum().backward()
        self.assertIsNone(points.grad)
        self.assertTrue(all(float(p.grad.abs().sum())>0 for p in grid.parameters()))

    def test_prior_detach_still_allows_renderer_to_move_gaussians(self):
        model=self.model()
        sum(v.sum() for v in model.prior(model.xyz)).backward()
        self.assertIsNone(model.xyz.grad)
        self.assertTrue(all(float(p.grad.abs().sum())>0 for p in model.hash_encoder.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.entropy_mlp.parameters()),0.)
        optimizer=model.make_optimizer()
        optimizer.zero_grad(set_to_none=True)
        model.update_learning_rates(optimizer,13)
        camera=Camera('probe',16,16,12.,12.,torch.eye(4))
        package=render(model,camera,13,'torch')
        loss,_=loss_terms(model,package,torch.zeros_like(package['image']),13)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.xyz.grad).all()))
        self.assertGreater(float(model.xyz.grad.abs().sum()),0.)
        before=model.xyz.detach().clone()
        optimizer.step()
        self.assertGreater(float((before-model.xyz).abs().max()),0.)

    def test_checkpoint_resumes_binary_tables_and_adam(self):
        model=self.model()
        optimizer=model.make_optimizer()
        sum(v.sum() for v in model.prior(model.xyz)).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.pt'
            save_checkpoint(path,model,optimizer,13)
            restored,opt,step,_=load_checkpoint(path,resume=True)
            self.assertEqual(step,13)
            for current,optim in ((model,optimizer),(restored,opt)):
                optim.zero_grad(set_to_none=True)
                sum(v.square().mean() for v in current.prior(current.xyz)).backward()
                optim.step()
            for name,parameter in model.hash_encoder.named_parameters():
                torch.testing.assert_close(parameter,dict(restored.hash_encoder.named_parameters())[name],atol=0,rtol=0)

    def test_float_grid_formats_are_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'float.pt'
            torch.save({'format':'smolgs-checkpoint-v2','config':{'hash_backend':'tcnn'}},path)
            with self.assertRaisesRegex(ValueError,'binary mixed hash requires v3'): load_checkpoint(path)
            path=Path(temporary)/'float.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('metadata.json',json.dumps({'format':'smolgs-archive-v5'}))
            with self.assertRaisesRegex(ValueError,'binary mixed hash requires v6'): decode_model(path)

    def _roundtrip(self,backend,device,codecs):
        model=self.model(backend,device)
        with torch.no_grad():
            model.features.copy_(torch.linspace(-3,3,model.features.numel(),device=device).reshape_as(model.features))
            # Zero must be exported as +1, matching BinarySTE during training.
            model.hash_encoder.xyz.table[0].zero_()
            model.freeze_coordinates()
            model.eval()
            original={n:p.detach().clone() for n,p in model.hash_encoder.named_parameters()}
            reference=quantized_hash_reference(model)
        with tempfile.TemporaryDirectory() as temporary:
            for codec in codecs:
                with self.subTest(codec=codec):
                    context_device='cpu' if codec=='cpu' else device
                    context=_canonical_model(reference,context_device)
                    with torch.no_grad():
                        _,_,_,_,qf,qs=context.prior(model.xyz.to(context_device))
                        expected_f=(torch.round(model.features.to(context_device)/qf)*qf).to(device)
                        expected_s=(torch.round(model.log_scales.exp().to(context_device)/qs)*qs).clamp_min(1e-7).to(device)
                    path=Path(temporary)/(codec+'.zip')
                    encode_model(model,path,codec,chunk_splats=3)
                    with zipfile.ZipFile(path) as archive:
                        meta=json.loads(archive.read('metadata.json'))
                        weights=load_torch(io.BytesIO(archive.read('networks.pt')))
                        self.assertEqual(meta['format'],'smolgs-archive-v6')
                        self.assertEqual(meta['hash_model'],'hac-binary-mixed-detached-v1')
                        self.assertEqual(len(meta['binary_hash']),4)
                        self.assertFalse(any(n.endswith('.table') for n in weights))
                        for name,info in meta['binary_hash'].items():
                            data=archive.read(info['file'])
                            self.assertEqual(len(data),(info['count']+7)//8)
                            bits=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little')[:info['count']]
                            np.testing.assert_array_equal(bits,(dict(model.state_dict())[name].cpu().numpy().reshape(-1)>=0).astype(np.uint8))
                    restored,_=decode_model(path,device)
                    torch.testing.assert_close(restored.features,expected_f,atol=0,rtol=0)
                    torch.testing.assert_close(restored.log_scales.exp(),expected_s,atol=1e-7,rtol=1e-6)
                    for name,parameter in model.hash_encoder.named_parameters():
                        torch.testing.assert_close(parameter,original[name],atol=0,rtol=0)
                        torch.testing.assert_close(dict(restored.hash_encoder.named_parameters())[name],dict(reference.hash_encoder.named_parameters())[name],atol=0,rtol=0)
                    camera=Camera('probe',16,16,12.,12.,torch.eye(4,device=device))
                    torch.testing.assert_close(render(reference,camera,25,'torch')['image'],render(restored,camera,25,'torch')['image'],atol=1e-6,rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_full_paper_grid_native_matches_torch_table_gradients(self):
        reference=MixedHashEncoder(Config(hash_backend='torch')).cuda()
        native=MixedHashEncoder(Config(hash_backend='cuda')).cuda()
        native.load_state_dict(reference.state_dict())
        points=torch.cat([torch.rand(29,3),torch.zeros(1,3),torch.ones(1,3),
                          torch.tensor([[-0.1,0.5,0.5],[1.1,0.5,0.5]])]).cuda().requires_grad_()
        expected,actual=reference(points),native(points)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        weights=torch.randn_like(expected)
        (expected*weights).sum().backward()
        (actual*weights).sum().backward()
        self.assertIsNone(points.grad)
        for name,table in reference.named_parameters():
            torch.testing.assert_close(dict(native.named_parameters())[name].grad,table.grad,atol=2e-6,rtol=2e-5)

    def test_cpu_bitpacked_roundtrip(self):
        self._roundtrip('torch','cpu',('cpu',))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_native_hash_cpu_and_cuda_arithmetic_roundtrip(self):
        self._roundtrip('cuda','cuda',('cpu','cuda'))


if __name__=='__main__':unittest.main()
