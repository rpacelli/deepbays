"""Edit the settings below, then run: python examples/explore_theory_multi.py

One-hot square-loss regression, or multi-class cross-entropy classification.
No arguments, plots, posterior sampling, or saved result files.
See docs/theory_examples.md for conventions and configuration recipes.
"""
from pathlib import Path
import sys
import time

import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepbays.rkgp import (FC_deep_multioutput, CNN_deep_multioutput,
                          FC_deep_classifier, CNN_deep_classifier)
from examples._theory_example import read_config, load_data, describe_model, evaluate, compare


# Switch to "cnn_multi.json" for a CNN. Edit architecture parameters in JSON.
MODEL_CONFIG = Path(__file__).with_name('model_configs') / 'mlp_multi.json'
LOSS = 'crossentropy'            # 'square' uses one-hot targets with the same D
DATA = dict(
    name='mnist',                # MLP/CNN: 'mnist'/'cifar10'; MLP: 'single_index'
    P=30, Ptest=60, seed=1234,
    input_dim=64,               # single_index vectors; ignored for image data
    side=28,                    # images only; use 32 for native RGB CIFAR10
    classes=[0, 1, 2],          # length must equal D in the model config
    root=None,                  # None -> ~/torchvision/<dataset>
    teacher_activation='erf',   # unused: class tasks bin a linear Gaussian index
)
THREADS = 1
VERBOSE = True
SOLVER = dict(maxiter=300, gtol=1e-6, n_restarts=0, random_state=42)
LAPLACE = dict(mode_tol=1e-10, mode_maxiter=100, max_dense_size=2000)
PREDICTION = dict(samples=1024, seed=42, batch_size=16)  # CE Sobol integration


def make_model(architecture, parameters, loss):
    parameters = dict(parameters)
    if parameters['D'] < 2:
        raise ValueError('use D >= 2 here; see explore_theory_single.py for scalar models')
    if loss == 'square':
        factory = FC_deep_multioutput if architecture == 'mlp' else CNN_deep_multioutput
        return factory(**parameters)
    if loss != 'crossentropy':
        raise ValueError("LOSS must be 'square' or 'crossentropy'")
    temperature = parameters.pop('T')
    if temperature <= 0:
        raise ValueError('cross-entropy requires T > 0')
    factory = FC_deep_classifier if architecture == 'mlp' else CNN_deep_classifier
    return factory(**parameters, beta=1/temperature, **LAPLACE, verbose=VERBOSE)


def main():
    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        started = time.perf_counter()
        architecture, parameters = read_config(MODEL_CONFIG)
        print(f'Config: {MODEL_CONFIG.name}; loss={LOSS}; parameters={parameters}', flush=True)
        X, Y, Xtest, Ytest, classes = load_data(DATA, architecture, parameters['D'], LOSS)
        model = make_model(architecture, parameters, LOSS)

        print('Preprocessing training kernels...', flush=True)
        start = time.perf_counter()
        model.preprocess(X, Y)
        print(f'  preprocessing: {time.perf_counter()-start:.2f}s', flush=True)
        describe_model(model)

        # One preprocessed model: the IW Laplace mode can warm-start EWA.
        model.setIW()
        iw = evaluate(model, 'IW', Xtest, Ytest, classes, LOSS, PREDICTION)

        print('EWA: optimizing Q...', flush=True)
        start = time.perf_counter()
        model.optimize(**SOLVER, verbose=VERBOSE)
        seconds = time.perf_counter()-start
        print(f'  optimization: {seconds:.2f}s; converged={model.converged}', flush=True)
        if not model.converged:
            raise RuntimeError('EWA did not converge; inspect the solver output before using predictions')
        ewa = evaluate(model, 'EWA', Xtest, Ytest, classes, LOSS, PREDICTION)
        ewa['optimization_seconds'] = seconds
        compare(iw, ewa, LOSS)
        print(f'Total (including data preparation): {time.perf_counter()-started:.2f}s', flush=True)
        return model, dict(IW=iw, EWA=ewa)


if __name__ == '__main__':
    model, results = main()  # Also available after running with python -i.
