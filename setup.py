from pdb import set_trace as T

from setuptools import find_packages, setup
from itertools import chain


setup(
    name="syllabus-rl",
    description="Syllabus Library"
    "Curriculum learning tools for RL",
    long_description_content_type="text/markdown",
    version="0.3",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pettingzoo==1.19.0',
        'supersuit==3.4.0',
        'gym==0.23.1',
        'numpy==1.23.3',
        'wandb>=0.15',
        'grpcio<=1.48.2',
        'gym-minigrid==0.0.6',
        'ray>=2.2.0',
        'torch>=2.0.1',
        'pygame>=2.5.1',
        'pymunk>=6.2.0',
        'nle==0.9.0',
        'procgen==0.10.7',
        'tensorboard>=2.13.0',
        'matplotlib>=3.7.1',
        'scipy==1.10.0',
    ],
    python_requires=">=3.8",
    author="Ryan Sullivan",
    author_email="rsulli@umd.edu",
    url="https://github.com/RyanNavillus/Syllabus",
    keywords=["Syllabus", "AI", "RL", "Curriculum learning"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

    ],
)

