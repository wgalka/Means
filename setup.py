from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='aggregationslib',
    packages=find_packages(include=['aggregationslib']),
    version='0.0.23',
    description='Python implementation of Arithmetic, quasi arithmetic and other aggregating functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='wgalka',
    license='TO DOO',

    url='https://github.com/wgalka/Means',
    download_url='https://github.com/wgalka/Means/archive/refs/tags/v0.0.22.tar.gz',

    install_requires=['numpy==1.22.2', 'pynverse~=0.1.4.4', 'Deprecated~=1.2.13'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)
