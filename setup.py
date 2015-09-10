from setuptools import setup, find_packages
from distutils.extension import Extension


version = (0, 0, 6)

setup(name='pyprophet-cli',
      version="%d.%d.%d" % version,
      author="Uwe Schmitt",
      author_email="uwe.schmitt@id.ethz.ch",
      description="tools for running pyoprohet on a computing cluster",
      license="BSD",
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
      zip_safe=False,
      install_requires=[
          "pyprophet>=0.15.1",
          "Click",
      ],
      entry_points={
          'console_scripts': [
              "pyprophet-cli=pyprophet_cli.main:cli",
              ]
      },
      )
