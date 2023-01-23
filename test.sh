#/bin/zsh

make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun4 --edition=2021
