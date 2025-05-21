from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='cvrmap',
    version='4.0.0',
    url='https://github.com/ln2t/CVRmap',
    author='Antonin Rovai',
    author_email='antonin.rovai@hubruxelles.be',
    description='Tools to compute maps of Cerebro-Vascular Reactivity',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'cvrmap = cvrmap.cli.cli:main',
        ]},
    include_package_data=True,
    package_data={}
)
