import glob
import shutil
import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pathlib import Path

# [文件名] = “路径”
module_path_map = {}
module_path_map["meter_infer"] = "."
module_path_map["deeplab"] = "modeling/"
module_path_map["aspp"] = "modeling/"
module_path_map["decoder"] = "modeling/"
# module_path_map["__init__"] = "modeling/sync_batchnorm/"
module_path_map["batchnorm"] = "modeling/sync_batchnorm/"
module_path_map["comm"] = "modeling/sync_batchnorm/"
module_path_map["replicate"] = "modeling/sync_batchnorm/"
module_path_map["unittest"] = "modeling/sync_batchnorm/"
# module_path_map["__init__"] = "modeling/backbone/"
module_path_map["drn"] = "modeling/backbone/"
module_path_map["mobilenet"] = "modeling/backbone/"
module_path_map["resnet"] = "modeling/backbone/"
module_path_map["xception"] = "modeling/backbone/"
module_path_map["custom_transforms"] = "dataloaders/"

ext_modules = []

for key in module_path_map:
    ext_modules.append(
        Extension(key, [str(Path(module_path_map[key]) / (key + '.py'))]))
    # print("******", key, Path(module_path_map[key]) / (key + '.py'), "*******")


setup(
    name='test cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules, compiler_directives={
                          'language_level': 3})
)

# 生成release
for file in glob.glob('*.pyd'):
    path_to_module = Path(module_path_map[file.split('.')[0]])
    print('Creating {}'.format(str(Path('release') / path_to_module)))
    (Path('release') / path_to_module).mkdir(parents=True, exist_ok=True)
    print('Copying {} to {}'.format(file, str(Path('release') / path_to_module)))
    shutil.copy(Path(file), (Path('release') / path_to_module))
    os.remove(Path(file))

shutil.copy(Path('modeling/__init__.py'),
            Path('release/modeling'))
shutil.copy(Path('modeling/sync_batchnorm/__init__.py'),
            Path('release/modeling/sync_batchnorm'))
shutil.copy(Path('modeling/backbone/__init__.py'),
            Path('release/modeling/backbone'))
print('Copying {} to {}'.format('modeling/__init__.py', str(Path('release'))))
shutil.copy(Path('main.py'), Path('release'))
(Path('release') / 'json').mkdir(parents=True, exist_ok=True)
shutil.copy(Path('json/mapping.json'), Path('release/json'))
shutil.copytree(Path('jit_functions'), Path('release/jit_functions'))
(Path('release') / 'run_model').mkdir(parents=True, exist_ok=True)
shutil.copy(Path('run/meter_seg_voc/deeplab-resnet/model_best.pth.tar'),
            Path('release/run_model'))
print('Copying {} to {}'.format('main.py, json, jit_function, model', str(Path('release'))))

# python compile.py build_ext --inplace
# cd release
# pyinstaller -F main.py

# 使用命令行 进行编译
# (应该也可以直接运行 setup.py 进行编译)
# 方式[1]　　python.exe setup.py build_ext --inplace　　此命令会生成 pyd 文件在 当前 dos 指向目录下
# 方式[2]　　python setup.py build　　此命令在当前dos 指向目录生成 .C 文件，还会生成 build 目录，并把 pyd 和 lib 文件 生成在该目录下
# 辅助命令　　python setup.py install　　复制安装模块到 python 目录 Lib\site-packages ，此命令一般在 方式[2] 之后配合使用。不是必须执行的。