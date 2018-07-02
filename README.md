# DocMT-MemNN

This is an implementation of our paper:

Sameen Maruf and Gholamreza Haffari: Document Context Neural Machine Translation using Memory Networks. ACL 2018.

Please cite our paper if you use our code.

# Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), using the development version (not release), e.g. 3.3.beta2 (http://bitbucket.org/eigen/eigen/get/3.3-beta2.tar.bz2)

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [boost](http://www.boost.org/), e.g., 1.58 using *libboost-all-dev* ubuntu package

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

# Building

First, clone the repository

git clone https://github.com/smar111/docmt-memnn.git

As mentioned above, you'll need the latest [development] version of eigen

hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb

A modified version of latest DyNet is already included (e.g., dynet folder). Please note that this is an older version and if the code is run with a new version some modifications may need to be made.

# CPU build

Compiling to execute on a CPU is as follows

mkdir build_cpu
cd build_cpu
cmake .. -DEIGEN3_INCLUDE_DIR=eigen [-DBoost_NO_BOOST_CMAKE=ON]
make -j 2

Boost note. The "-DBoost_NO_BOOST_CMAKE=ON" can be optional but if you have a trouble of boost-related build error(s), adding it will help to overcome. 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN [-DBoost_NO_BOOST_CMAKE=ON] -DMKL=TRUE -DMKL_ROOT=MKL

substituting in different paths to EIGEN and MKL if you have placed them in different directories. 

This will build the 3 binaries
    
build_cpu/src/docmt-memnn
build_cpu/src/sentrnnlm


