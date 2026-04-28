from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements(file_name: str):
    requirements_path = Path(__file__).resolve().parent / file_name
    requirements = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        requirements.append(item)
    return requirements

setup(
    name="chitu-diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=_read_requirements("requirements.txt"),
    package_dir={
        'chitu_core': 'chitu_core',
        'chitu_diffusion': 'chitu_diffusion'
    }
)