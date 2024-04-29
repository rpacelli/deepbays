from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'Bayesian deep neural networks in the proportional regime'
LONG_DESCRIPTION = 'Compute expected predictor of a Bayesian deep neural networks in the proportional regime in a non-parametric way using the equivalent GP'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name='deepbays', 
        version=VERSION,
        author='Rosalba Pacelli',
        author_email='<rosalba.pacelli@gmail.com>',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'torch', 'torchvision', 'scipy'],#, 'tensorflow'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'machine learning', 'gaussian processes', 'neural networks']
)