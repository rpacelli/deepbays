"""Sample an FC/CNN posterior from NPZ data, or run a small synthetic example.

NPZ fields: X, y (training), optionally Z, z (test inputs and targets).
Run --help or see docs/sampling.md. Outputs include memory-mapped trajectories,
chain diagnostics and predictive summaries. No kernel/theory calculation is needed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deepbays.NN import FCNet, ConvNet
from deepbays.sampling import (NetworkPosterior, SamplerConfig, sample_posterior,
                              classification_statistics, logit_contrasts, diagnostics_and_mcse)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--network',choices=['fc','cnn'],default='fc')
    parser.add_argument('--task',choices=['regression','classification'],default='classification')
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--outputs',type=int,default=2)
    parser.add_argument('--activation',choices=['id','erf','relu'],default='erf')
    parser.add_argument('--precisions',type=float,nargs='+',default=[1.])
    parser.add_argument('--gamma',type=float,default=1.)
    parser.add_argument('--mask',type=int,nargs='+',default=[2])
    parser.add_argument('--stride',type=int,nargs='+',default=[2])
    parser.add_argument('--padding',nargs='+',default=['valid'],choices=['valid','same'])
    parser.add_argument('--temperature',type=float,default=.2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dtype',choices=['float64','float32'],default='float64')
    parser.add_argument('--sampler',choices=['hmc','nuts'],default='nuts')
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--draws',type=int,default=500)
    parser.add_argument('--chains',type=int,default=4)
    parser.add_argument('--chain-offset',type=int,default=0)
    parser.add_argument('--seed',type=int,default=173)
    parser.add_argument('--target-accept',type=float,default=.85)
    parser.add_argument('--max-tree-depth',type=int,default=7)
    parser.add_argument('--trajectory-length',type=float,default=1.)
    parser.add_argument('--likelihood-batch-size',type=int)
    parser.add_argument('--predict-batch-size',type=int,help='default: one full prediction batch')
    parser.add_argument('--checkpoint-batches',action='store_true',help='recompute chunk activations to save memory')
    parser.add_argument('--buffer-size',type=int,default=16)
    parser.add_argument('--data-on-cpu',action='store_true',help='keep training and prediction inputs on CPU')
    parser.add_argument('--save-weights',action='store_true')
    parser.add_argument('--compile',action='store_true',help='compile the potential and prediction callable')
    parser.add_argument('--compile-mode',choices=['default','max-autotune-no-cudagraphs'],default='default',
                        help='torch.compile mode, used with --compile')
    parser.add_argument('--cudnn-benchmark',action='store_true',help='autotune deterministic CUDA convolutions')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--diagnostic-points',type=int,default=16)
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('runs/pyro_posterior'))
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    if args.draws<4 or args.diagnostic_points<1 or args.threads<1:
        parser.error('draws>=4, diagnostic-points>=1 and threads>=1 are required')
    torch.set_num_threads(args.threads)
    dtype=getattr(torch,args.dtype)
    if args.data is None:
        rng=np.random.default_rng(args.seed)
        shape=(8,) if args.network=='fc' else (1,8,8)
        inputs=rng.normal(size=(36,*shape))
        teacher=rng.normal(size=(int(np.prod(shape)),args.outputs))/np.sqrt(np.prod(shape))
        signal=inputs.reshape(36,-1)@teacher
        if args.task=='classification':
            from scipy.special import softmax
            targets=np.array([rng.choice(args.outputs,p=row) for row in softmax(signal,axis=-1)])
        else:
            targets=signal+np.sqrt(args.temperature)*rng.normal(size=signal.shape)
        X,y,Z,z=inputs[:20],targets[:20],inputs[20:],targets[20:]
    else:
        with np.load(args.data,allow_pickle=False) as data:
            X,y=data['X'],data['y']
            Z=data['Z'] if 'Z' in data else X
            z=data['z'] if 'z' in data else (y if 'Z' not in data else None)
    priors=args.precisions*(args.layers+1) if len(args.precisions)==1 else args.precisions
    if len(priors)!=args.layers+1:
        parser.error('provide one precision or layers+1 precisions')
    common=dict(act=args.activation,precisions=priors,gamma=args.gamma,D=args.outputs)
    if args.network=='fc':
        if X.ndim!=2:parser.error('FC inputs must have shape (P,input_dimension)')
        builder=FCNet(X.shape[1],args.width,args.layers,**common)
    else:
        if X.ndim!=4:parser.error('CNN inputs must have shape (P,channels,height,width)')
        single=lambda value:value[0] if len(value)==1 else value
        builder=ConvNet(X.shape[-2:],args.width,args.layers,inputChannels=X.shape[1],
                         mask=single(args.mask),stride=single(args.stride),padding=single(args.padding),**common)
    posterior=NetworkPosterior(builder.Sequential(),X,y,priors=priors,task=args.task,
                               temperature=args.temperature,device=args.device,dtype=dtype,
                               batch_size=args.likelihood_batch_size,
                               data_device='cpu' if args.data_on_cpu else None,
                               checkpoint_batches=args.checkpoint_batches)
    prediction_inputs=torch.as_tensor(Z,dtype=posterior.dtype,
                                      device='cpu' if args.data_on_cpu else posterior.device)
    def predict(theta):
        return posterior.predict(theta,prediction_inputs,args.predict_batch_size)
    if args.compile:
        posterior.compile(mode=args.compile_mode)
        predict=torch.compile(predict,mode=args.compile_mode)
    config=SamplerConfig(sampler=args.sampler,chains=args.chains,warmup=args.warmup,draws=args.draws,
                         seed=args.seed,chain_offset=args.chain_offset,target_accept=args.target_accept,
                         max_tree_depth=args.max_tree_depth,trajectory_length=args.trajectory_length,
                         progress=args.progress,cudnn_benchmark=args.cudnn_benchmark)
    args.output.mkdir(parents=True,exist_ok=True)
    settings=vars(args).copy();settings['data']=str(args.data) if args.data else None;settings['output']=str(args.output)
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    # Archive exactly the inputs used for this experiment, also for synthetic runs.
    np.savez_compressed(args.output/'data.npz',X=X,y=y,Z=Z,**({} if z is None else {'z':z}))
    def finished(result,chain):
        row=result.chain_info[chain-args.chain_offset]
        print(f'chain {chain}: {row["seconds"]:.1f}s; acceptance={row["acceptance"]:.3f}; '
              f'divergences={row["divergences"]}',flush=True)
    result=sample_posterior(posterior,config,save_weights=args.save_weights,
                            observables={'predictions':predict},
                            output_dir=args.output/'samples',buffer_size=args.buffer_size,on_chain_end=finished)
    predictions=result.get_samples('predictions')
    # Diagnose a documented subset to keep ArviZ memory bounded for large test sets.
    subset=predictions[:,:,:args.diagnostic_points]
    latent=logit_contrasts(subset) if args.task=='classification' else np.asarray(subset)
    centered=latent-latent.mean((0,1))
    observables=dict(latent=latent,centered_second_moment=centered**2)
    metrics={}
    if args.task=='classification':
        summary=classification_statistics(predictions,z)
        np.savez_compressed(args.output/'classification.npz',**summary)
        if z is not None:
            metrics={k:summary[k] for k in ('mean_predictor_accuracy','posterior_sample_accuracy')}
            observables['accuracy']=summary['accuracy_draws']
    else:
        # Only the selected diagnostic points are summarized here; all predictions
        # remain in the store for subsequent analysis without holding weights on GPU.
        np.savez_compressed(args.output/'regression_subset.npz',mean=latent.mean((0,1)),variance=latent.var((0,1)))
    diagnostics,mcse=diagnostics_and_mcse(observables,chain_info=result.chain_info)
    np.savez_compressed(args.output/'mcse.npz',**mcse)
    report=dict(metrics=metrics,diagnostics=diagnostics,diagnostic_points=min(args.diagnostic_points,len(Z)))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Diagnostic gate: {"passed" if diagnostics["passed"] else "inconclusive"}; {metrics}',flush=True)
    print(f'Saved {args.output}',flush=True)

if __name__=='__main__':
    main()
