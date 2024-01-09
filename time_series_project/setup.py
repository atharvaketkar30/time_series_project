from setuptools import setup, find_packages
from typing import List

def get_requirements(path: str) -> List[str]:
    '''
    This finds all the requirements in the requirements.txt file
    '''

    dont_include = '-e .'
    requirements = []
    with open(path, 'r') as file:
        reqs = file.readlines()
        reqs = [req.replace('\n', '') for req in reqs]

        if dont_include in reqs:
            reqs.remove(dont_include)
    
    return reqs

setup(
    name='time_series_project',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    author='Atharva Ketkar',
    install_requires= get_requirements('requirements.txt')
)