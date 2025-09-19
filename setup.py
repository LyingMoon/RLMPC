from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rlmpc",
    version="1.0.0",
    author="Xiaolong Jia, Nikhil Bajaj",
    author_email="xij62@stanford.edu",
    description="Predictive RL for MPC: Adapting to Model Parameter Variations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LyingMoon/RLMPC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "cvxpy>=1.2.0",
        "osqp>=0.6.0",
        "gym>=0.21.0",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
)