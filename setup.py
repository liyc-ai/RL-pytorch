import itertools

from setuptools import find_packages, setup


def get_version():
    """Gets the imitation_base version."""
    path = "ilkit/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(
    name="ilkit",
    version=get_version(),
    description="A clean code base for imitation learning and reinforcement learning.",
    author="Yi-Chen Li",
    author_email="ychenli.X@gmail.com",
    url="https://github.com/BepfCp/ilkit",
    packages=find_packages(include=["ilkit*"]),
    python_requires=">=3.7",
    install_requires=[
        "gym==0.22.0",
        "gymnasium[all]",
        "h5py",
        "hydra-core==1.3.2",
        "nni",
        "numba",
        "numpy==1.23.5",
        "omegaconf==2.3.0",
        "setuptools==65.5.1",
        "stable_baselines3",
        "torch",
        "tqdm"
    ]
)
