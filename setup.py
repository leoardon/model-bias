import logging
import os
import re
import shutil
import subprocess
import sys
from codecs import open
from distutils.command.register import register as register_orig
from distutils.command.upload import upload as upload_orig

from setuptools import find_packages, setup

NAME = "model_bias"


def _get_version():
    with open(os.path.join(NAME, "__init__.py")) as fp:
        return re.match(r"__version__\s*=\s*\"(?P<version>.*)\"", fp.read()).group(
            "version"
        )


def _get_long_description():
    with open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
        encoding="utf-8",
    ) as readme_file:
        long_description = readme_file.read()
    return long_description


def _get_requirements():
    with open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt"),
        encoding="utf-8",
    ) as requirements_file:
        requirements = [
            l.strip()
            for l in requirements_file.readlines()
            if not (l.strip().startswith("#") or l.strip().startswith("-"))
        ]
    return requirements


def _build_docker_image(client, force=False):
    try:
        _img = client.images.get(NAME)
        if force:
            raise docker.errors.ImageNotFound("", "", "")
    except docker.errors.ImageNotFound:
        logging.warning("Building %s docker image.", NAME)
        out = client.api.build(
            path=os.path.abspath(os.path.dirname(__file__)),
            dockerfile=os.path.join(os.path.abspath(os.path.dirname(__file__)), "Dockerfile"),
            rm=True,
            tag=NAME,
            decode=True
        )
        for line in out:
            if "stream" in line:
                sys.stdout.write(line["stream"])

class RegisterCmd(register_orig):
    def _get_rc_file(self):
        return os.path.join(".", ".pypirc")


class UploadCmd(upload_orig):
    def _get_rc_file(self):
        return os.path.join(".", ".pypirc")


if sys.argv[-1] == "build":
    subprocess.call(["python3", "setup.py", "bdist_wheel"])
    sys.exit()

if sys.argv[-1] == "publish":
    subprocess.call(["python3", "setup.py", "sdist", "upload", "-r", "local"])
    subprocess.call(["python3", "setup.py", "bdist_wheel", "upload", "-r", "local"])
    sys.exit()

if sys.argv[-1] == "clean":
    shutil.rmtree(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist"), True
    )
    shutil.rmtree(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"), True
    )
    shutil.rmtree(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "%s.egg-info" % NAME),
        True,
    )
    sys.exit()

if sys.argv[-1] == "tag":
    VERSION = _get_version()
    subprocess.call(["git", "tag", "-a", VERSION, "-m", "'version %s'" % VERSION])
    subprocess.call(["git", "push", "--tags"])
    sys.exit()

if sys.argv[-1] == "test" or sys.argv[-2] == "test":
    try:
        import docker
        client = docker.from_env()

        _build_docker_image(client, sys.argv[-1] == "--force-build")

        out = client.containers.run(
            image=NAME,
            command=("-m pytest -p no:cacheprovider --disable-warnings --cov-fail-under 1 "
                     "--cov-report term --cov-report html --cov=. tests/"),
            name=NAME,
            remove=True,
            detach=False,
            stdout=True,
            stream=True,
            volumes={
                os.path.abspath(os.path.dirname(__file__)): {'bind': '/code/', 'mode': 'rw'}
            }
        )
        for l in out:
            sys.stdout.buffer.write(l)

    except ImportError:
        logging.error("You must have docker python package installed. Run 'pip install docker'")

    sys.exit()

setup(
    name=NAME,
    version=_get_version(),
    description="Model Bias Package",
    long_description=_get_long_description(),
    author="Leo Ardon",
    classifiers=[
        "Development Status :: 5 - Production",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="bias model",
    include_package_data=True,
    packages=find_packages(),
    install_requires=_get_requirements(),
    test_suite=f"{NAME}.tests.test_all",
    cmdclass={"register": RegisterCmd, "upload": UploadCmd},
)
