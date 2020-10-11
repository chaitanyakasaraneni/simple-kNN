from pathlib import Path
import re
import setuptools

if __name__ == "__main__":

    # Read description from README
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()
        
    # Read metadata from version.py
    with Path("simple_kNN/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))

    setuptools.setup(name='simple_kNN',
            version=metadata["version"],
            author=metadata["author"],
            author_email='kc.kasaraneni@gmail.com',
            description='Simple kNN algorithm with k-Fold Cross Validation',
            install_requires=["numpy"],
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/chaitanyakasaraneni/simple-kNN",
            packages=['simple_kNN'],
            package_dir={'simple_kNN':'simple_kNN'},
            package_data={'simple_kNN': ['dataset/data/*.csv', 'dataset/data/*.csv.gz']},
            classifiers=[
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Intended Audience :: Science/Research",
            ],
            python_requires='>=3.6',
            zip_safe=False)
