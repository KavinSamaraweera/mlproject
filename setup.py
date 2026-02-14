#this will find all the tools installed in entire ML app
from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT ='-e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    This function will return the list of requirements
    '''
    requiremnts = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        # here when we read a line, / will be get added, so it needs to be blank
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)


    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Kavindu',
    author_email='kavinsamaraweera@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)