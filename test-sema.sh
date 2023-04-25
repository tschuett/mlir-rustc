#/bin/zsh

#make -j && ./tools/rustc/rustc --path=`pwd`/examples/fun1 --edition=2021

make rustc && tools/rustc/rustc --fwith-sema --crate-name=foo --path=testsuite/withsema/Struct.rs || echo failed
