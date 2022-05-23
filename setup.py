"""Packaging"""

from setuptools import setup

setup(
    name='ersa',
    version='0.0.0',
    description='A method for analyzing and extrapolating random search.',
    long_description=open('README.rst', 'r').read(),
    url='https://github.com/nalourie/ersa',
    author='Nicholas Lourie',
    author_email='dev@nicholaslourie.com',
    keywords='ersa extrapolated random search analysis hyper-parameter'
             ' tuning artificial intelligence ai machine learning ml',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='',
    packages=['ersa', 'experiments'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[ln.strip() for ln in open('requirements.txt', 'r')],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>= 3.9',
    zip_safe=False,
)
