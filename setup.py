from pathlib import Path

from setuptools import find_packages, setup

version = '0.0.253'

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='aggregationslib',
    packages=find_packages(include=['aggregationslib']),
    version=version,
    description='Python implementation of Arithmetic, quasi arithmetic and other aggregating functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='wgalka',
    license='TO DOO',

    url='https://github.com/wgalka/Means',
    download_url=f'https://github.com/wgalka/Means/archive/refs/tags/v{version}.tar.gz',

    install_requires=["numpy>=1.23.2"],  # Python 3.11 minimum
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)
