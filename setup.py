from setuptools import find_packages, setup

setup(
    name="SpineDiffusion",
    version="0.1.0",
    url="https://gitlab.ethz.ch/4d-spine/spinediffusion",
    author="Pau Altur Pastor",
    author_email="paltur@student.ethz.ch",
    description="SpineDiffusion: A diffusion model for synthethic 3D backscan generation with Internal Spinal Line as prior information.",
    packages=find_packages(),
    install_requires=["pytorch_lightning==2.2.5", "diffusers==0.28.2"],
)
