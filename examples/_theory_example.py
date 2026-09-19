"""Shared data preparation and terminal summaries for the two editable examples.

Model construction, preprocessing and solver calls are in the example scripts.
This is example support code, not an additional deepbays API.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from deepbays import tasks


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['architecture'] not in ('mlp', 'cnn'):
        raise ValueError("architecture must be 'mlp' or 'cnn'")
    return config['architecture'], config['parameters']


def load_data(settings, architecture, outputs, loss):
    """Return X, Y, Xtest, Ytest, test class indices (or None for regression).

    Image labels use the order of settings['classes'], even if a small training
    subset omits a class. Scalar square-loss targets are -1/+1; multi-output
    square-loss targets are one-hot. Cross-entropy targets are integer indices.
    """
    if loss not in ('square', 'crossentropy'):
        raise ValueError("LOSS must be 'square' or 'crossentropy'")
    p, pt = settings['P'], settings['Ptest']
    if p < 2 or pt < 1:
        raise ValueError('use P >= 2 and Ptest >= 1')
    name, seed = settings['name'], settings['seed']
    classes = 2 if outputs == 1 else outputs
    scalar_regression = outputs == 1 and loss == 'square'
    print(f'Loading {name}: P={p}, Ptest={pt}, seed={seed}', flush=True)
    start = time.perf_counter()
    if name == 'single_index':
        if architecture != 'mlp':
            raise ValueError("single_index is a vector task for MLPs; use mnist or cifar10 for a CNN")
        # Keep the native vector inputs. For class tasks, bin a linear Gaussian
        # teacher index into equiprobable classes; this is an example-level
        # label construction, not a spatial or convolutional teacher.
        activation = settings['teacher_activation'] if scalar_regression else 'id'
        task = tasks.single_index_dataset(settings['input_dim'], activation, dataSeed=seed)
        X, y, Z, z = task.make_data(p, pt)
        if not scalar_regression:
            boundaries = ndtri(np.arange(1, classes) / classes)
            y = np.digitize(np.asarray(y).ravel(), boundaries)
            z = np.digitize(np.asarray(z).ravel(), boundaries)
    elif name in ('mnist', 'cifar10'):
        side = settings['side']
        labels = settings['classes']
        if len(labels) != classes or len(set(labels)) != classes:
            raise ValueError(f'select exactly {classes} distinct classes for this model')
        root = Path(settings['root'] or f'~/torchvision/{name}').expanduser()
        factory = tasks.mnist_dataset if name == 'mnist' else tasks.cifar_dataset
        task = factory(side**2, labels, dataSeed=seed, dirpath=str(root))
        options = dict(flatten=False, normalize=True)
        if name == 'cifar10':
            options['grayscale'] = False
        # Existing task loaders download missing files and normalize using only
        # the selected training inputs; the official train/test split is kept.
        X, y, Z, z = task.make_data(p, pt, **options)
        if len(X) != p or len(Z) != pt:
            raise ValueError('requested more examples than the selected classes contain')
        mapping = {label: index for index, label in enumerate(labels)}
        y = np.array([mapping[int(label)] for label in np.asarray(y).ravel()])
        z = np.array([mapping[int(label)] for label in np.asarray(z).ravel()])
    else:
        raise ValueError("DATA['name'] must be 'single_index', 'mnist' or 'cifar10'")

    X, Z = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    if architecture == 'mlp':
        X, Z = X.reshape(p, -1), Z.reshape(pt, -1)
    if name == 'single_index' and scalar_regression:
        Y, Ytest = np.asarray(y, dtype=float).ravel(), np.asarray(z, dtype=float).ravel()
        test_classes = None
    else:
        y, z = np.asarray(y, dtype=int).ravel(), np.asarray(z, dtype=int).ravel()
        test_classes = z
        if loss == 'crossentropy':
            Y, Ytest = y, z
        elif outputs == 1:
            Y, Ytest = 2. * y - 1., 2. * z - 1.
        else:
            Y, Ytest = np.eye(outputs)[y], np.eye(outputs)[z]
        print(f'  train/test class counts: {np.bincount(y, minlength=classes)} / '
              f'{np.bincount(z, minlength=classes)}', flush=True)
    print(f'  data ready in {time.perf_counter()-start:.2f}s; X={X.shape}, Y={Y.shape}', flush=True)
    return X, Y, Z, Ytest, test_classes


def describe_model(model):
    print(f'Model: {type(model).__name__}, L={model.L}, width={model.N1}', flush=True)
    shapes = getattr(model, 'patch_shapes', None)
    if shapes is None and hasattr(model, 'features'):
        shapes = getattr(model.features, 'patch_shapes', None)
    if shapes is not None:
        print(f'  hidden patch grids: {shapes}; final patches={model.d}', flush=True)


def evaluate(model, method, Xtest, Ytest, test_classes, loss, prediction):
    """Predict once, print a compact report, and return arrays for manual checks."""
    Q = np.array(model.optQ, copy=True)
    if Q.ndim == 1:
        print(f'{method}: layer q={Q}; product Q={np.prod(Q):.6g}', flush=True)
    else:
        eigenvalues = np.linalg.eigvalsh(Q)
        print(f'{method}: Q shape={Q.shape}, eigenvalue min/median/max='
              f'{eigenvalues.min():.6g}/{np.median(eigenvalues):.6g}/{eigenvalues.max():.6g}', flush=True)
    print(f'{method}: predicting {len(Xtest)} test examples...', flush=True)
    start = time.perf_counter()
    if loss == 'crossentropy':
        values = model.predict_statistics(Xtest, **prediction)
        p = values['probabilities']
        selected = np.arange(len(Ytest)), Ytest
        metrics = dict(
            mean_probability_accuracy=float(np.mean(p.argmax(1) == Ytest)),
            expected_sample_accuracy=float(values['argmax_probabilities'][selected].mean()),
            predictive_nll=float(-np.log(np.maximum(p[selected], np.finfo(float).tiny)).mean()),
            mean_probability_variance=float(values['probability_variance'].mean()))
        state = model.laplace_state
        print(f'  Laplace mode: steps={state.iterations}, residual={state.mode_residual:.3g}', flush=True)
    else:
        # Scalar FC's predict() has no batch-size keyword. All square-loss
        # classes cache the quantities needed by averageLoss(). Scalar targets
        # stay one-dimensional to avoid broadcasting in the legacy FC routine.
        mean = np.asarray(model.predict(Xtest))
        sample_mse, mean_mse, variance = model.averageLoss(Ytest)
        if hasattr(model, 'predictive_variance'):
            latent_variance = model.predictive_variance
        else:
            latent_variance = model.rK0L - np.sum(model.K0_invK * model.rK0XL, axis=1)
        values = dict(mean=mean, variance=np.array(latent_variance, copy=True))
        metrics = dict(mean_predictor_mse=float(mean_mse), expected_sample_mse=float(sample_mse),
                       mean_latent_variance=float(variance))
        if test_classes is not None:
            classes = (mean.ravel() > 0).astype(int) if mean.ndim == 1 else mean.argmax(1)
            metrics['mean_output_accuracy'] = float(np.mean(classes == test_classes))
    seconds = time.perf_counter() - start
    for key, value in metrics.items():
        print(f'  {key}: {value:.6g}', flush=True)
    print(f'  prediction (including posterior fitting/integration): {seconds:.2f}s', flush=True)
    return dict(Q=Q, prediction=values, metrics=metrics, prediction_seconds=seconds)


def compare(iw, ewa, loss):
    key = 'probabilities' if loss == 'crossentropy' else 'mean'
    delta = ewa['prediction'][key] - iw['prediction'][key]
    print(f'EWA vs IW {key}: RMS difference={np.sqrt(np.mean(delta**2)):.6g}, '
          f'max absolute difference={np.max(np.abs(delta)):.6g}', flush=True)
