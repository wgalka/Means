from pathlib import Path

from setuptools import find_packages, setup

version = '0.0.3'

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    classifiers=[
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    name='aggregationslib',
    packages=find_packages(include=['aggregationslib', 'aggregationslib.*']),
    # version=version,
    use_scm_version=True,
    description='Python implementation of Arithmetic, quasi arithmetic and other aggregating functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.12',
    author='wgalka',
    license='TO DOO',

    url='https://github.com/wgalka/Means',
    download_url=f'https://github.com/wgalka/Means/archive/refs/tags/v{version}.tar.gz',

    install_requires=["numpy>=2.2.2",
                      "pynverse"],
    setup_requires=['pytest-runner>=6.0.1'],
    tests_require=['pytest==8.3.4'],
    extras_require={
        'test': ['pytest>=8.3.4'],
    },
    test_suite='tests',
)
