from setuptools import setup, find_packages

setup(
    name='gym_goal',
    version='0.0.1',
    description='Goal domain OpenAI Gym environment',
    author='Craig James Bester',
    packages=find_packages(include=['gym_goal', 'gym_goal.*']),
    install_requires=['gym',
                      'pygame',  # 'pygame>=1.9.3'
                      'numpy',   # 'numpy>=1.14.0'
    ]
) 
