"""Install this Python package.
"""

# pylint: disable=invalid-name

import re
import os.path
from setuptools import setup, find_packages


class Setup():
    """Convenience wrapper (for C.I. purposes) of the `setup()` call form `setuptools`.
    """
    def __init__(self, **kw):
        self.conf = kw
        self.work_dir = os.path.abspath(os.path.dirname(__file__))

        # Automatically fill `package_data` from `MANIFEST.in`. No need to repeat lists twice
        assert "package_data" not in self.conf
        assert "include_package_data" not in self.conf
        package_data = {}
        with open(os.path.join(self.work_dir, "MANIFEST.in"), encoding="utf-8") as fp:
            for line in fp.readlines():
                line = line.strip()
                m = re.search(r"include\s+(.+)/([^/]+)", line)
                assert m
                module = m.group(1).replace("/", ".")
                fileName = m.group(2)
                if module not in package_data:
                    package_data[module] = []
                package_data[module].append(fileName)
        if package_data:
            self.conf["include_package_data"] = True
            self.conf["package_data"] = package_data

        # Automatically fill the long description from `README.md`. Filter out lines that look like
        # "badges".
        assert "long_description" not in self.conf
        assert "long_description_content_type" not in self.conf
        with open(os.path.join(self.work_dir, "README.md"), encoding="utf-8") as fp:
            ld = "\n".join([row for row in fp if not row.startswith("[![")])
        self.conf["long_description"] = ld
        self.conf["long_description_content_type"] = "text/markdown"

    def __str__(self):
        return str(self.conf)

    def __call__(self):
        setup(**self.conf)


SETUP = Setup(
    name="hipe4ml",

    # LAST-TAG is a placeholder. Automatically replaced at deploy time with the right tag
    version="0.0.15",

    description="Minimal heavy ion physics environment for Machine Learning",

    url="https://github.com/hipe4ml/hipe4ml",
    author="hipe4ml-developers",
    author_email="hipe4ml@googlegroups.com",
    license="GPL",

    # See https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],

    # What does your project relate to?
    keywords="",

    # You can just specify the packages manually here if your project is simple. Or you can use
    # find_packages().
    packages=find_packages(exclude=['tutorials']),

    # List run-time dependencies here. These will be installed by pip when your project is
    # installed. For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["uproot>=4.2.2", "matplotlib>=3.5.1", "pandas>=1.3.5", "scikit-learn>=1.0.2",
                      "xgboost>=1.5.2,<2.0", "lightgbm>=3.3.2", "shap==0.42.1", "pyarrow>=7.0.0",
                      "ipython>=7.32.0", "optuna>=2.10.0", "numpy<1.25"],

    python_requires=">=3.8",

    # List additional groups of dependencies here (e.g. development dependencies). You can install
    # these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": ["pylint>=2.6.2", "flake8>=3.8.4", "pytest>=6.2.2", "twine>=3.3.0", "setuptools>=53.0.0",
                "wheel>=0.36.2"]
    },

    # Although 'package_data' is the preferred approach, in some case you may need to place data
    # files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    data_files=[],

    # To provide executable scripts, use entry points in preference to the "scripts" keyword. Entry
    # points provide cross-platform support and allow pip to create the appropriate form of
    # executable for the target platform. See:
    # https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
    entry_points={
        "console_scripts": ["hipe4ml = hipe4ml:entrypoint"]
    }
)

if __name__ == "__main__":
    SETUP()
