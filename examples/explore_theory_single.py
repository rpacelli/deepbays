"""Edit the settings below, then run: python examples/explore_theory_single.py

Scalar square-loss regression, or binary cross-entropy (one logit contrast).
No arguments, plots, posterior sampling, or saved result files.
See docs/theory_examples.md for conventions and configuration recipes.
"""
from pathlib import Path
import sys
import time

import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepbays.rkgp import FC_deep_vanilla, CNN_deep, FC_deep_classifier, CNN_deep_classifier
from examples._theory_example import read_config, load_data, describe_model, evaluate, compare


# For a CNN, select "cnn_single.json" and set DATA['name'] to mnist/cifar10.
# Edit architecture parameters in JSON; inputs follow the selected dataset.
MODEL_CONFIG = Path(__file__).with_name('model_configs') / 'cnn_single.json'
LOSS = 'square'                  # 'square' or 'crossentropy'
DATA = dict(
    name='cifar10',         # MLP only; 'mnist'/'cifar10' support MLP and CNN
    P=50, Ptest=200, seed=1234,
    input_dim=64,               # single_index vector dimension; need not be square
    side=32,                    # images only: use 28 for MNIST, 32 for CIFAR10
    classes=[0, 1],             # image classes in this order
    root=None,                  # None -> ~/torchvision/<dataset>
    teacher_activation='erf',   # only for scalar Gaussian regression
)
THREADS = 1
VERBOSE = True
SOLVER = dict(maxiter=300, gtol=1e-6, n_restarts=0, random_state=42)
SCALAR_SOLVER = dict(Qmin=1e-2, Qmax=1e3)  # FC square: bracket for layer q
LAPLACE = dict(mode_tol=1e-10, mode_maxiter=100, max_dense_size=2000)
PREDICTION = dict(samples=1024, seed=42, batch_size=16)  # CE Sobol integration


def make_model(architecture, parameters, loss):
    parameters = dict(parameters)
    if loss == 'square':
        factory = FC_deep_vanilla if architecture == 'mlp' else CNN_deep
        return factory(**parameters)
    if loss != 'crossentropy':
        raise ValueError("LOSS must be 'square' or 'crossentropy'")
    temperature = parameters.pop('T')
    if temperature <= 0:
        raise ValueError('cross-entropy requires T > 0')
    # D=2 logits have one orthonormal contrast; this is the existing two-logit
    # prior, not an independently normalized one-logit sigmoid network.
    factory = FC_deep_classifier if architecture == 'mlp' else CNN_deep_classifier
    return factory(**parameters, D=2, beta=1/temperature, **LAPLACE, verbose=VERBOSE)


def main():
    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        started = time.perf_counter()
        architecture, parameters = read_config(MODEL_CONFIG)
        print(f'Config: {MODEL_CONFIG.name}; loss={LOSS}; parameters={parameters}', flush=True)
        X, Y, Xtest, Ytest, classes = load_data(DATA, architecture, 1, LOSS)
        model = make_model(architecture, parameters, LOSS)

        print('Preprocessing training kernels...', flush=True)
        start = time.perf_counter()
        # The legacy FC routine expects torch targets; keep them 1D.
        targets = torch.as_tensor(Y, dtype=torch.float64) if isinstance(model, FC_deep_vanilla) else Y
        model.preprocess(X, targets)
        print(f'  preprocessing: {time.perf_counter()-start:.2f}s', flush=True)
        describe_model(model)

        model.setIW()
        iw = evaluate(model, 'IW', Xtest, Ytest, classes, LOSS, PREDICTION)

        print('EWA: optimizing Q...', flush=True)
        start = time.perf_counter()
        if isinstance(model, FC_deep_vanilla):
            model.optimize_ArcsinhLog(**SCALAR_SOLVER, verbose=VERBOSE)
        else:
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
