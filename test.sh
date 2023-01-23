#/bin/zsh

make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun3 --edition=2021
