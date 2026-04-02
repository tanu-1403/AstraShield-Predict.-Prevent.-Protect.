from setuptools import setup, find_packages

setup(
    name="astrashield",
    version="2.0.0",
    author="AstraShield Team",
    description="Predict. Prevent. Protect. — Autonomous Orbital Debris Intelligence System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-team/astrashield",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "httpx>=0.25.0",
    ],
    entry_points={
        "console_scripts": [
            "astrashield=main:run",
            "astrashield-api=api.server:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
