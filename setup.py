from setuptools import setup, find_packages

setup(
    name='osc-transformer-relevance-detector',
    version='1.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'osc-transformer-relevance-detector = cli_package.osc_transformer_relevance_detector.main:app',
        ],
    },
)
