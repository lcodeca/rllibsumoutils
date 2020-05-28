#!/usr/bin/env python3

""" PYTHON3 - Setup for rllibsumoutils.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

from setuptools import setup

def readme():
    """ README """
    with open('README.md') as freader:
        return freader.read()

setup(name='rllibsumoutils',
      version='0.1',
      description='Python3 interface for RLLIB and SUMO simulator.',
      url='http://github.com/lcodeca/rllibsumoutils',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)',
          'Programming Language :: Python :: 3 :: Only',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
      ],
      author='Lara Codeca',
      author_email='lara.codeca@gmail.com',
      license='EPL-2.0',
      packages=['rllibsumoutils'],
      install_requires=['lxml', 'rtree'],
      include_package_data=True,
      zip_safe=False)
