"""Packaging"""

from setuptools import setup

setup(
    name='opda',
    version='0.1.0',
    description='A framework for the design and analysis of deep learning'
                ' experiments.',
    long_description=open('README.rst', 'r').read(),
    url='https://github.com/nalourie/opda',
    author='Nicholas Lourie',
    author_email='dev@nicholaslourie.com',
    keywords='opda optimal design analysis hyperparameter hyper-parameter'
             ' tuning machine learning ml deep learning dl'
             ' artificial intelligence ai',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache-2.0',
    packages=['opda', 'experiments'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[ln.strip() for ln in open('requirements.txt', 'r')],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>= 3.8',
    zip_safe=False,
)
