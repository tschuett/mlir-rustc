#/bin/zsh

make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun1 --edition=2021
