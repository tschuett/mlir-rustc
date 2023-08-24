# ThinLTO next steps

ThinLTO

With FatLTO, Clang puts LLVM IR into the object files. The linker
reads the IR of all object files and merges them into one
llvm::Module. Afterwards the optimizer runs over the module. It runs
with only thread and can consume a lot of memory, but it can run a
variety of optimizations across TUs.

ThinLTO addresses some of the limitations of FatLTO. From the start it
is designed for distributed build system, ala Bazel. Alternatively,
LLD will starts threads.

## Easy with CMake

For LLVM:
```shell
 ... -DLLVM_ENABLE_LTO=Thin
```

or in pure CMake

```shell
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
```

## First phase

ThinLTO compiles the source code to object files including LLVM IR and a summary.

```shell
clang++ -flto=thin -O3 -c -o interesting.o interesting.cpp
clang++ -flto=thin -O3 -c -o exciting.o  exciting.cpp
```

## Thin link

In the thin link, LLD reads **only** the summaries, performance analysis, and writes index files.

```shell
clang++ -flto=thin -O3 -o thinlto.objects  -Wl,-plugin-opt,thinlto-index-only=thinlto.objects interesting.o exciting.o
```

## Optimizations exploiting the summaries

In the third phase, the LLVM IR in the objects is compiled to AArch64 exploiting the index files.

```shell
clang++ -c -x ir interisting.o -O3 -o interisting-native.o -fthinlto-index=interisting.o.thinlto.bc
clang++ -c -x ir exciting.o -O3 -o exciting-native.o -fthinlto-index=exciting.o.thinlto.bc
```

## Final Link

The final step is merging the AArch64 object files into an executable file.

```shell
> clang++ -o interisting-native.o executable exciting-native.o
```

The first and third phase are embarrassingly parallel and can be
distributed over a Linux cluster in the basement. The thin link is a
serialization point, but it is supposed to be short resp. thin.

# Cross-TU inlining

In-tree ThinLTO sends function bodies over the network. Later on they
can be imported and inlined on the remote workers.

# Whole Program Devirtualization

xxx

# Function specialization

By sending constants and mangled functions names over the network, we
could perform remote function specialization on the worker
nodes. There will be more specialization candidates than before. The
current function specialization pass is limited to functions in the
same TU.

# Outlining

By sending hashes over the network, we could implement more aggressive
outlining and function merging. Potential users are embedded and
mobile application.

# Two-phase offload programming models

One object file with AArch64 IR and OpenMP-ish summary and one object
file with NVPTX IR and OpenMP-ish summary. Can we do a *partial*
ThinLTO for optimizations between host and device code, e.g., constant
propagation?

Not sure anymore.

# MLIR LTO

The MLIR C/C++ Frontend Working Group chatted about MLIR-LTO:
"(Thin-)LTO for MLIR very interesting for GPUs (Microsoft)". This
would require some kind of mlir-thin-linker.

Flang could be a first customer. It would probably need a
flang-thin-linker to handle its dialects and domain knowledge.

# Final notes

There will be more RFCs and there will be no Pull Requests this week.

We will work first on function specialization with ThinLTO. You can
expect more RFCs.

We want you to help to innovate new optimizations with ThinLTO. We
want to make it easier resp. more natural to put facts into the
summaries. Maybe even having a tutorial for writing facts into the
summary.

# BibTex
- ThinLTO: Scalable and incremental LTO
- Scalable Size Inliner for Mobile Applications
- Improving Machine Outliner for ThinLTO: also Nikolai Tillmann
- Practical Global Merge Function with ThinLTO  (EuroLLVM 2023)
- https://discourse.llvm.org/t/thinlto-import-functions-to-enable-function-specialization-in-thinlto/58627
- https://discourse.llvm.org/t/rfc-safer-whole-program-class-hierarchy-analysis/65144
- https://lists.llvm.org/pipermail/llvm-dev/2019-December/137543.html
- https://reviews.llvm.org/D93838
- https://docs.google.com/document/d/1iS0_4q7icTuVK6PPnH3D_9XmdcrgZq6Xv2171nS4Ztw/edit#heading=h.oggqptmb1frj
