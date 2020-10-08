import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='simple_kNN',
      version='0.5',
      description='Simple kNN algorithm with k-Fold Cross Validation',
      author='Chaitanya Krishna Kasaraneni',
      author_email='kc.kasaraneni@gmail.com',
      long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaitanyakasaraneni/simple-kNN",
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        zip_safe=False)
