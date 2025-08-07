from setuptools import setup, find_packages

setup(
    name='pure_rl_ucav',
    version='1.0.0',
    description='Pure Reinforcement Learning for UCAV Dogfight',
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "torchaudio>=0.13.0",
        "matplotlib>=3.7.5",
        "pandas>=2.0.3",
        "gym>=0.26.2",
        "gymnasium>=0.26.2",
        "pyyaml>=6.0.2",
        "tensorboard>=2.13.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "rltorch",  # For SAC memory components
    ],
    python_requires='>=3.8',
    author='TUBITAK Project Team',
    author_email='your.email@domain.com',
    url='https://github.com/your-username/pure-rl-ucav',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='reinforcement-learning, autonomous-systems, air-combat, td3, sac',
    project_urls={
        'Bug Reports': 'https://github.com/your-username/pure-rl-ucav/issues',
        'Source': 'https://github.com/your-username/pure-rl-ucav',
    },
)