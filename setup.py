import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('pybest', 'version.py')

with open(ver_file) as f:
    exec(f.read())

# Long description will go up on the pypi page
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    REQUIRES = f.readlines()

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    requires=REQUIRES,
    entry_points={
        'console_scripts': [
            'pybest=pybest.cli:main',
            'pybest_view_surf=pybest.utils:view_surf',
            'pybest_npy2mgz=pybest.utils:pybest_npy2mgz',
            'pybest_vol2surf=pybest.utils:pybest_vol2surf'
            ]
        }
)

if __name__ == '__main__':
    setup(**opts)
