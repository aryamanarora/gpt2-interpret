from setuptools import setup


install_requires = [
    "torch",
    "transformers",
    "tqdm",
    "matplotlib",
    "numpy"
]


setup(
	name="gptutils",
	install_requires=install_requires,
	version="0.1",
	scripts=[],
	packages=['gptutils']
)