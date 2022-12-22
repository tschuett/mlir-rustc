#!/bin/bash

rm CMakeCache.txt

cmake -DLLVM_DIR=/usr/local/opt/llvm  .
