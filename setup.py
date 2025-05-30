from setuptools import setup, find_packages

setup(
    name="matryoshka-wrapper",
    version="0.1.0",
    description="Wrapper for PEFT-based matryoshka embeddings with projection heads.",
    author="Abdulrahman Kamel",
    author_email="your-email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft"
    ],
    python_requires='>=3.7',
)
