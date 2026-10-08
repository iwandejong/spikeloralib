import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spikeloralib",
    version="0.2.0",
    author="Iwan de Jong, Arné Schreuder, Anna S. Bosman (SpikeLoRA); "
           "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (original LoRA/loralib)",
    author_email="idj.idejong@gmail.com",
    description="PyTorch implementation of low-rank adaptation (LoRA) and SpikeLoRA, a spiking "
                 "variant that gates LoRA's A-matrix activations with a leaky integrate-and-fire "
                 "neuron to learn activation sparsity in the low-rank space.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iwandejong/spikeloralib",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "tiktoken",
        "protobuf",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
