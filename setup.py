from setuptools import find_packages, setup

setup(
    name='aggregationslib',
    packages=find_packages(include=['aggregationslib']),
    version='0.0.1',
    description='Python implementation of Arithmetic, quasi arithmetic and other aggregating functions',
    author='wgalka',
    license='TO DOO',

    install_requires=['numpy==1.22.2', 'pynverse~=0.1.4.4', 'Deprecated~=1.2.13'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
