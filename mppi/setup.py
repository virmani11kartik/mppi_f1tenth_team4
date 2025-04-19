from setuptools import setup

package_name = 'mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['mppi/config/config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='f1tenth mppi',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mppi_node = mppi.mppi_node:main',
        ],
    },
)

