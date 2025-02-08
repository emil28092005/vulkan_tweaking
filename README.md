# Advanced Computer Graphics in Game Development

A template repository for Advanced Computer Graphics in Game Development course.

## Pre-requirements

- Version control: [Git](https://git-scm.com)
- Build system: [Cmake](https://cmake.org)
- C++17 compatible compiler:
  - MSVC on Windows
  - Clang on MacOS
  - GCC on Linux
- Code editor: [VSCode](https://code.visualstudio.com/Download)
- [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)

## How to build the project

Use `git clone --recursive` to clone the repo with submodules or run `git submodule update --init --recursive` after the first clone.

Go to the project folder and run the next command:

```shell
mkdir build
cd build
cmake ..
cmake --build .
```

## How to compile shaders

On Windows:

```bat
compile_shaders.bat
```

On Linux or MacOS:

```shell
./compile_shaders.sh
```

## How to run

```shell
build/acg_in_gd_lab
```

## Third-party tools and data

- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) (Apache 2.0, MIT licenses)
- [GLFW](https://www.glfw.org) (Zlib license)
- [GLM](https://glm.g-truc.net) (Modified MIT license)
- [Cxxopts](https://github.com/jarro2783/cxxopts) (MIT license)
- [STB](https://github.com/nothings/stb) (Public domain, MIT licenses)
- [TinyGLTF](https://github.com/syoyo/tinygltf) (MIT license)
- [glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) (CC-BY 4.0 International)
