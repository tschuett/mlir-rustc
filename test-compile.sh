#/bin/zsh

#make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun1 --edition=2021

make rustc && tools/rustc/rustc --fcompile --crate-name=foo --path=testsuite/compile/Function.rs || echo failed
