
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='gpcsd', 
    version='1.0.2', 
    description='Gaussian process current source density estimation', 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/natalieklein/gpcsd', 
    author='Natalie Klein',  
    author_email='natalie.elizabeth.klein@gmail.com', 
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'), 
    python_requires='>=3.7, <4',
    install_requires=['autograd',
                      'h5py>=3.2.1', 
                      'matplotlib',
                      'networkx>=2.6.2',
                      'numpy>=1.20.3',
                      'scikit-image>=0.18.1',
                      'scipy>=1.6.2',
                      'tqdm',
                      'joblib',
                     ],  
    project_urls={  
         'Paper': 'https://arxiv.org/abs/2104.10070',
    },
)