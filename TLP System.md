# TLP System

1. System Requirements

   - Ubuntu 18.04
   - Python 3.8
   - At least 130 GB disk space
   - An adequate GPU

2. Software Requirements

   ```bash
   sudo apt-get update &&
   sudo apt-get install wget software-properties-common &&
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test &&
   wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - &&
   sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" &&
   sudo apt-get update
   ```

   ```bash
   sudo apt-get install build-essential clang-8 lld-8 g++-7 cmake ninja-build libvulkan1 python python-pip python-dev python3-dev python3-pip libpng-dev libtiff5-dev libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev git
   ```

   ```bash
   sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180 &&
   sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180
   ```

   ```bash
   pip3 install --user -Iv setuptools==47.3.1
   pip3 install --user distro
   pip3 install --user wheel auditwheel
   ```

3. Copy the folders of compiled `UnrealEngine` and `carla`

4. Recompile `UnrealEngine`

   ```bash
   cd path/to/UnrealEngine_4.26
   make clean
   make
   ```

5. Set Unreal Engine environment variable

   ```bash
   export UE4_ROOT=path/to/UnrealEngine_4.26
   ```

   or

   ```bash
   gedit ~/.bashrc
   ```

   add following line at the bottom of the file

   ```bash
   export UE4_ROOT=path/to/UnrealEngine_4.26
   ```

6. Recompile `carla`

   ```bash
   cd path/to/carla_origin
   make clean
   make PythonAPI
   make launch
   ```

7. Install `carla` package

   Search for `whl` file compiled in step 6.

   ```bash
   pip3 install <path/to/wheel>.whl
   ```

8. Start the simulation

   Press `Play` to start simulation

