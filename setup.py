import pathlib

import setuptools

setuptools.setup(
    name="discqua",
    version="0.0.1",
    description="Discussion Quality Aspects",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/nikos-Gk/dialogue-quality-measures",
    author="Nikos Gkoumas",
    author_email="n.goumas@athenarc.gr",
    licence="Apache-2.0 license",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10,<=3.12",
    install_requires=[
        "pandas >=1.5.0",
        "convokit==3.5.0",
        "openai==0.28",
        # "llama-cpp-python==0.3.9",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
