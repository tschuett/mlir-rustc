#!/bin/bash

rm CMakeCache.txt

cmake -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir .
