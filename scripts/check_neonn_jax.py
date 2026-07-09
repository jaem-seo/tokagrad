#!/usr/bin/env python
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import jax, jax.numpy as jnp
from tokagrad.neonn_jax import load_neonn_ensemble, predict_neonn_jax, neonn_jax_status

p=argparse.ArgumentParser()
p.add_argument('--model-dir', default='external_models/neural')
p.add_argument('--model-name', default='jbsnn')
p.add_argument('--max-nets', type=int, default=1)
args=p.parse_args()
ok,msg=neonn_jax_status(args.model_dir,args.model_name,args.max_nets)
print(msg)
if not ok: raise SystemExit(2)
ens=load_neonn_ensemble(args.model_dir,args.model_name,args.max_nets)
print('input_names =', ens.input_names)
print('output_names=', ens.output_names)
x=jnp.asarray(ens.nets[0].scale_mean_in)
y=predict_neonn_jax(x,args.model_dir,args.model_name,args.max_nets)
J=jax.jacobian(lambda z: predict_neonn_jax(z,args.model_dir,args.model_name,args.max_nets))(x)
print('sample output =', y)
print('dy/dx shape =', J.shape)
print('finite output:', bool(jnp.all(jnp.isfinite(y))))
print('finite jacobian:', bool(jnp.all(jnp.isfinite(J))))
