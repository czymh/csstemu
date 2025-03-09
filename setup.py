from setuptools import setup
import os
import re

def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "CEmulator/__init__.py")
    with open(init_path, 'r') as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

packages = ['CEmulator',
            'CEmulator/emulator',
            'CEmulator/GaussianProcess',
            'CEmulator/hankl',]

setup(
    name='CEmulator',
    version=get_version(),
    author='Zhao Chen',
    author_email='chyiru@sjtu.edu.cn',
    description='A python package for CSST cosmological emulator, which can predict various cosmological statistics within miliseconds.',
    packages=packages,
    url='https://github.com/czymh/csstemu',
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={'CEmulator': ['data/*']},
    include_package_data=True,
)
