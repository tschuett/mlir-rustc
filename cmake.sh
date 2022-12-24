#!/bin/bash

rm CMakeCache.txt

cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/usr/local/opt/llvm  .
