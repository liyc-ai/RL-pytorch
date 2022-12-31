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


def read_requirements():
    # To generate requirements:
    # pipreqs . --ignore [ignored dir] --force
    with open("./requirements.txt", "r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()
    return install_requires


extras = {"nni": ["nni"]}

extras["all"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], extras.keys())))
)

requires = read_requirements()

setup(
    name="ilkit",
    version=get_version(),
    description="A clean code base for imitation learning and reinforcement learning.",
    author="Yi-Chen Li",
    author_email="ychenli.X@gmail.com",
    url="https://github.com/BepfCp/ilkit",
    packages=find_packages(include=["ilkit.*"]),
    install_requires=requires,
    extras_require=extras,  # pip install -e ".[all]"
)
