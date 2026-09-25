from setuptools import setup, find_packages
from codecs import open
from os import path


from mrrd import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='mrrd',
      version=__version__,
      description='Diffusion for Long-Horizon Multi-Robot Path Planning in Human-Shared Environments',
      author='Vaibhav Sanjay',
      author_email='vsanjay@andrew.cmu.edu',
      packages=find_packages(where=''),
      install_requires=requires_list,
      )
