from setuptools import setup, find_packages

def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='SegRunLib',
    version='0.0.4',
    author='msst',
    author_email='mihailshutov105@gmail.com',
    description='This is the module for segmentator running',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/NotYourLady',
    packages=['SegRunLib', 'SegRunLib.ml', 'SegRunLib.scripts'],
    install_requires=['torch',
                      'torchio',
                      'nibabel',
                      'tqdm'],
    classifiers=[
    'Programming Language :: Python :: 3.10',
    'Operating System :: OS Independent'
    ],
    keywords='ml nn cnn',
    project_urls={
    'GitHub': 'https://github.com/NotYourLady'
    },
    python_requires='>=3.6'
)
