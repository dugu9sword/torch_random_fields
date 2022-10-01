from setuptools import setup, find_packages

setup(
    name="torch_random_fields",
    version="1.0",
    keywords=["torch", "CRF", "PGM"],
    description="eds sdk",
    license="WTFPL Licence",

    url="https://github.com/dugu9sword/torch_random_fields",
    author="dugu9sword",
    author_email="dugu9sword@163.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,

    scripts=[],
    entry_points={}
)