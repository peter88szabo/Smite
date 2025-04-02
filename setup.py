from setuptools import setup, find_packages

setup(
    name="Smite",  # Name of your project
    version="0.1.0",  # Initial version
    author="Peter Szabo",  # Replace with your name
    author_email="peter88szabo@gmail.com",  # Replace with your email
    description="Quasiclassical trajectory method for ab-initio molecular dynamics with semiclassically quantized initial conditions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/peter88szabo/Smite",  # Replace with your repository URL
    license="GNU",  # Replace with the appropriate license
    packages=find_packages(where="src"),  # Find all packages in the src directory
    py_modules=["smite"],
    package_dir={"": "src"},  # Root is src
    python_requires=">=3.6",  # Specify your minimum Python version
    install_requires=[
        "numpy>=1.21.0",
        # Add your dependencies here, e.g., "numpy>=1.20.0"
    ],

    extras_require={
        "pyscf": ["pyscf"],  # Optional dependency for PySCF API
        "sparrow": ["scine-sparrow"],  # Optional dependency for scine-sparrow API
        "all": ["pyscf", "scine-sparrow"],  # Install all optional dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)

