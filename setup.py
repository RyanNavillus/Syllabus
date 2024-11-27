from setuptools import find_packages, setup


extras = dict()
extras['test'] = ['pytest>=8.1.1', 'pytest-benchmark>=3.4.1', 'cmake', 'ninja', 'gym',
                  'nle>=0.9.0', 'matplotlib>=3.7.1', 'pygame', 'pymunk', 'scipy>=1.10.0', 'tensorboard>=2.13.0', 'shimmy']
extras['docs'] = ['sphinx-tabs', 'sphinxcontrib-spelling', 'furo']
extras['all'] = extras['test'] + extras['docs']

setup(
    name="syllabus-rl",
    description="Protable curricula for RL agents",
    long_description_content_type="text/markdown",
    version="0.6",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gymnasium>=0.28.0',
        'numpy>=1.24.0',
        'torch>=2.0.1',
        'pettingzoo',
        'joblib',
        'ray',
    ],
    extras_require=extras,
    python_requires=">=3.8",
    author="Ryan Sullivan",
    author_email="rsulli@umd.edu",
    url="https://github.com/RyanNavillus/Syllabus",
    keywords=["Syllabus", "AI", "RL", "Curriculum Learning", "Unsupervised Environment Design"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.11",
    ],
)
