from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'docopt',
    'pandas',
    'tensorflow',
    'keras',
    'scikit-learn',
    'filterpy',
]

setup(
  name='src',
  version='0.1',
  author = 'Vladislav Stasenko',
  author_email = 'vladislav.stasenko@fh-swf.de',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='An example package for training on Cloud ML Engine.'
)