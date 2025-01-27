from setuptools import find_packages,setup
from typing import List
Hyphen_e='-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function returns reqired required packages'''
    requirements=[]
    with open(file_path, 'r') as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if Hyphen_e  in requirements:
            requirements.remove(Hyphen_e)

    return requirements

setup(
name="gmlproject",
version="0.0.1",
author="gopi",
author_email="n210204@rguktn.ac.in",
packages=find_packages(),
install_requirements=get_requirements('Requirements.txt')

)