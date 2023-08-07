from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='aggregationslib',
    packages=find_packages(include=['aggregationslib']),
    version='0.0.26',
    description='Python implementation of Arithmetic, quasi arithmetic and other aggregating functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='wgalka',
    license='TO DOO',

    url='https://github.com/wgalka/Means',
    download_url='https://github.com/wgalka/Means/archive/refs/tags/v0.0.26.tar.gz',

    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)
