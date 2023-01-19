#/bin/zsh

make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun2 --edition=2021
