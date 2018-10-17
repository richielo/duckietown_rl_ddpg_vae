from setuptools import setup

setup(name='duckietown_rl',
      version='1.0',
      install_requires=['gym>=0.5',
                        'gym_duckietown', # _agent
                        'sklearn',
                        'torch',
                        'numpy',
                        'matplotlib',
                        'scipy']
      )
