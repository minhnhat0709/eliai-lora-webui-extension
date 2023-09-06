from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("eliai_auth",  ["eliai_auth.py"]),
    Extension("lora_decryption",  ["lora_decryption.py"]),
    Extension("networks_eliai",  ["networks_eliai.py"]),
#   ... all your modules that need be compiled ...
]
setup(
    name = 'My Program Name',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)