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
