#!/bin/bash

rm CMakeCache.txt

cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/usr/local/opt/llvm  .
