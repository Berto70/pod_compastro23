from setuptools import setup, Extension

#To get the version
exec(open('fireworks/version.py').read())
#To get the long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name = "fireworks",
    version =__version__,
    author = "Giuliano Iorio",
    author_email = "giuliano.iorio.astro@gmail.com",
    description = "N-body playground",
    license = "MIT",
    packages=['fireworks','fireworks/nbodylib'],
    long_description=long_description,
    python_requires='>=3.7',
    install_requires=['numpy',"pyfalcon","pytest"]
)