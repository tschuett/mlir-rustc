RFC: VPlan Analysis I

# Introduction

The cannonical 5 nested for-loops. What kind of VPLan native
analysises do we need for efficient and correct *auto* vectorization?

```c
for( unknown ) {
  for( unknown ) {
    for( huge ) {
      for( unknown ) {
        for( small ) {
        }
        for( huge ) {
          a[i] = b[i-1] + b[i] + b[i+1]
        }
      }
    }
    for( huge ) {
      for( unknown ) {
        for( small ) {
        }
        for( huge ) {
          if (likely) {
          } else if (maybe) {
          } else {
          }
        }
      }
    }
  }
}
```

Diego contributed the [DominatorTree](https://reviews.llvm.org/D48815)
and [LoopInfo](https://reviews.llvm.org/D48816). Unfortunately,
LoopInfo disappeared.

# Dominator Tree

It allows to reason about dominance between various IR objects. The
generic implementation abstracts away over the basic block types. It
supports currently VPlan, LLVM IR, and MLIR. The generic dominator
construction uses the Semi-NCA algorithm instead of Lengauer-Tarjan.

MLIR uses the `DominanceInfo` class to reason about dominance between
blocks, operations, and values.

# LoopInfo

It analyzes loop nests of natural loops. The implementation abstracts
away over the basic block type and uses a dominator tree for the
analysis.

```c
for (unsigned i = 0; i < x; ++i) {
  for (unsigned j = 0; j < y; ++j {
  }
  for (unsigned k = 0; k < z; ++k) {
  }
}

for (unsigned l = 0; l < a; ++l) {
}
```

The generic `analyze` function takes a generic dominator tree. Thus,
it works for functions and VPlans.

MLIR has `CFGLoop` on top of the generic LoopInfo class.

# Scalar Evolution aka Chains of recurrences

It helps to reason about induction variables, back edges, and trip
counts (DominatorTree and LoopInfo). The current implementation
supports LLVM IR.

```c
for (unsigned i = 0; i < 10; ++i) {
  a[i] = 5.0 * b[i];
  c += 42 + c*i;
}
```

The VPlan variant will differ. In the first phase, it will be tailored
to the needs of the vectorizer and not for generic loop optimizations.

SCEV I supports SCEVConstant, a constant integer value. Add support
for floating points?

The MLIR story is a bit of an unknown. Can SCEV be implemented on top
of the `Arith` dialect? Furthermore, is the investment worth it?

# Memory Access Analysis

What is the maximum save vectorization width? Where is dependency,
consecutive access, and gather/scatters. May I vectorize with scalable
vectors?

```c
for(unsigned i = 0; i < unknown; ++i) {
  a[i] = b[i]
  c[i] = d[i-1] + d[i] + d[i+1]
  e[i] = f[i] + 5.0 * g[i]
  h[i] = (h[i-1] + h[i] + h[i+1]) / 3.0
}
```

The analysis relies on DominatorTree, LoopInfo, AliasAnalysis, and
SCEV.

# Divergence Analysis

It helps you to reason about divergence in control flow. Can I
vectorize the loop with predication? Can I transform the plan into
something more desirable? Which branches are divergent?

```c
for(unsigned i = 0; i < unknown; ++i) {
  if (i > unknown2) {
  } else {
  }
}
```

# Cost models

Is plan `A` or plan `B` more desirable? Is vectorization with scalar,
NEON, SVE, SVE2, or SME more desirable? What is outer-loop
vectorization for the 5 nested for loops?

Outer vectorize the 5 nested for loops or try tiling?

```c
def NVIDIAGrace {
  string Name = "nvidia-grace";
  string NrOfSockets = "2";
  string NrOfCoresPerSocket = "72";
  string L1CacheData = "64KB";
  string L1CacheCode = "64KB";
  string L2Cache = "1MB";
  string L3Cache = "234MB";
}
```

# Caching

During VPlan to VPlan transformations, the planner can query
analysises about the current plan and manage perseverance for the new
plan, i.e., `PreservedAnalyses`.

```c
DivergenceAnalysis& divergence = AM.get<DivergenceAnalysis>(plan);
```

# Alias analysis

customer: Memory Access Analysis

# Missed analysis

Unknown?

* Reduction detection?
* Idom Detection?
* Vector Function Database?

# Testing Strategy

New `llvm-vplan-tester` tool:
* read IR
* generate first VPlan
** `-mtriple=aarch64-apple-ios -mattr=+sme2`
** `--print-memory-access-analysis`
** `--print-transform-predicate`
** `--print-divergence-analysis`
** `--print-cost-model`
** `--print-scev`

Support out-of-tree playground experiments:
* move VPlan into its own pass decoupled from the inner loop vectorizer
* move VPlan headers into a top include directory
** also needed by `llvm-vplan-tester`

# Alternatives Considered

Wait for 5 more years

# Near Future

I have in my
[playground](https://github.com/tschuett/vplan-analysis/tree/main) a
second LoopInfo on top of VPlan native with plans to upstream.

[1] Chains of recurrences—a method to expedite the evaluation of closed-form functions.
    Olaf Bachmann, Paul S. Wang, and Eugene V. Zima.
[2] J. Absar “Scalar Evolution - Demystified”: https://www.youtube.com/watch?v=AmjliNp0_00
[3] Matrices https://riscv.org/blog/2023/02/xuantie-matrix-multiply-extension-instructions/
[4] A Review - Loop Dependence Analysis for Parallelizing Compiler
[5] IBM https://www.cs.rice.edu/~vs3/PDF/ibmjrd97.pdf
[6] VPlan https://llvm.org/docs/Proposals/VectorizationPlan.html
[7] System model https://reviews.llvm.org/D58736
    https://github.com/aws/aws-graviton-getting-started
[8] A Unified Semantic Approach for the Vectorization and Parallelization of Generalized Reductions
[9] Outer-loop vectorization: revisited for short SIMD architectures
[10] GCC SCEV https://github.com/gcc-mirror/gcc/blob/master/gcc/tree-scalar-evolution.cc
[11] Divergence Analysis https://discourse.llvm.org/t/rfc-a-new-divergence-analysis-for-llvm/48686
[12] Divergence Analysis https://reviews.llvm.org/D50433

# Oddities

How can I calculate VPlan-SCEV after five transforms? I need to find
constants in the VPlan IR!

```
VPValue(const unsigned char SC, Value *UV = nullptr, VPDef *Def = nullptr);

// DESIGN PRINCIPLE: Access to the underlying IR must be strictly limited to
// the front-end and back-end of VPlan so that the middle-end is as
// independent as possible of the underlying IR. We grant access to the
// underlying IR using friendship. In that way, we should be able to use VPlan
// for multiple underlying IRs (Polly?) by providing a new VPlan front-end,
// back-end and analysis information for the new IR.
```

VPInstruction: how to identify VScale intrinsic when Opcode == Instruction::Call?

* [phi nodes](https://llvm.org/docs/LangRef.html#phi-instruction)
* [block arguments](https://github.com/apple/swift/blob/main/docs/SIL.rst#basic-blocks)
* [block arguments](https://mlir.llvm.org/doxygen/classmlir_1_1Block.html)

* icc had amazing vectorization reports (file on the disk).  Use
@llvm.dbg.* to find functions, loops, variables with data dependencies
(names) and their line numbers.

* IR for failed vectorization attempts.
