from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="matryoshka-embeddings",
    version="0.1.0",
    author="Abdalrahman54",
    author_email="your.email@example.com",
    description="A library for creating Matryoshka embeddings with multiple dimensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/matryoshka-embeddings",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="embeddings, matryoshka, transformers, peft, nlp, arabic",
    project_urls={
        "Bug Reports": "https://github.com/your-username/matryoshka-embeddings/issues",
        "Source": "https://github.com/your-username/matryoshka-embeddings",
    },
)
