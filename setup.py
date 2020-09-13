from setuptools import find_packages, setup

# Get the long description from the relevant file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PyStatDP',
    version='0.0.1',
    description='Counterexample Detection Using Statistical Methods for Incorrect Differential-Privacy Algorithms.',
    long_description=long_description,
    url='https://github.com/OpenMined/PyStatDP',
    author='',
    author_email='',
    license='MIT',
    classifiers=[
        'Development Status ::  Alpha',
        'Intended Audience :: Developers :: Researchers',
        'Topic :: Differential Privacy :: Statistics',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='Differential Privacy, Hypothesis Test, Statistics',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.5',
    install_requires=['numpy', 'tqdm', 'numba', 'jsonpickle', 'python-dp'],
    extras_require={
        'test': ['pytest-cov', 'pytest', 'coverage', 'flaky', 'scipy'],
    },
    entry_points={
        'console_scripts': [
            'PyStatdp=statdp.__main__:main',
        ],
    },
)
