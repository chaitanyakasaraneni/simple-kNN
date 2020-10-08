import setuptools

setuptools.setup(name='simple_kNN',
      version='0.1',
      description='Simple kNN algorithm with k-Fold Cross Validation',
      author='Chaitanya Krishna Kasaraneni',
      author_email='kc.kasaraneni@gmail.com',
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        zip_safe=False)