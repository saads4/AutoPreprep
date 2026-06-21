from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


setup(
    name="autopreprocessor",
    version="0.1.0",
    description="Automatic tabular preprocessing with RAG-assisted encoding selection.",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "autopreprocessor": [
            "chroma_db/*",
            "chroma_db/*/*",
        ],
    },
    install_requires=REQUIREMENTS,
    python_requires=">=3.9",
)
