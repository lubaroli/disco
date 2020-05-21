import setuptools

with open("LICENSE", "r") as fh:
    license = fh.read()

with open("README.md", "r") as fh:
    readme = fh.read()

short_description = (
    "This package provides a collection of modules and examples to quickly "
    "setup experiments using DISCO: Double Likelihood-free Inference "
    "Stochastic Control."
)
setuptools.setup(
    name="disco-rl",
    version="0.0.1",
    author="Lucas Barcelos",
    author_email="lucas.barcelos@sydney.edu.au",
    description=short_description,
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/lubaroli/disco",
    packages=setuptools.find_packages(exclude=("notebooks", "docs", "scripts")),
    license="GPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gym >= 0.12.4",
        "numpy >= 1.18.1",
        "matplotlib >= 3.1.2",
        "scipy >= 1.4.1",
        "torch >= 1.3",
    ]
)
