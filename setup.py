from setuptools import setup, find_packages

setup(
    name='CEmulator',
    version='0.1.0',
    author='Zhao Chen',
    author_email='chyiru@sjtu.edu.cn',
    description='A python package for CSST cosmological emulator, which can predict various cosmological statistics within miliseconds.',
    packages=find_packages(),
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
)
