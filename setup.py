from setuptools import setup, find_packages

VERSION = '0.0.9' 
DESCRIPTION = 'Bayesian deep neural networks in the proportional regime as renormalized kernel gaussian processes'
LONG_DESCRIPTION = 'Compute expected predictor of a Bayesian deep neural networks in the proportional regime in a non-parametric way using the equivalent GP'

setup(
        name='deepbays', 
        version=VERSION,
        author='Rosalba Pacelli',
        author_email='<rosalba.pacelli@gmail.com>',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'torch', 'torchvision', 'scipy'],
        
        keywords=['python', 'machine learning', 'gaussian processes', 'neural networks']
)