from setuptools import find_packages, setup


extras = dict()
extras['test'] = ['cmake', 'ninja', 'nle>=0.9.0', 'matplotlib>=3.7.1', 'scipy==1.10.0', 'tensorboard>=2.13.0']
extras['docs'] = ['sphinx-tabs', 'sphinxcontrib-spelling', 'furo']
extras['all'] = extras['test'] + extras['docs']

setup(
    name="syllabus-rl",
    description="Syllabus Library"
    "Curriculum learning tools for RL",
    long_description_content_type="text/markdown",
    version="0.4",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gymnasium>=0.28.0',
        'numpy>=1.26.0',
        'ray[rllib]>=2.8.1',
        'torch>=2.0.1',
    ],
    extras_require=extras,
    python_requires=">=3.8, <=3.11",
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
