Pre-RFC: The LoopPlaner

# Loop Optimization

The cannonical 5 nested for-loops. What is the optimal transformation
that minimizes execution time?

```c
for( unknown ) {
  for( unknown ) {
    for( huge ) {
      for( unknown ) {
        for( small ) {
        }
        for( huge ) {
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

Flatten the inner three loops? Unroll known small loops? Unswitch the
ifs. Blindly run the loop passes (gcc or Clang?) against it? ...
Probably none of them!

GCC and LLVM both use loop passes to optimize loops. They both suffer
from pass-ordering issues. In contrast VPlan and SLP both rely on abstraction
raising, evaluate several plans using cost models, and only write back to
IR if they found a better solution.

# The LoopPlaner

The LoopPlanner follows the design of VPlan. It reads the IR, performs
legality checks and analysis, and transforms it into a higher
abstraction representation. It will try every kind of loop
optimization but refuses to vectorize to optimize the loop
nest. Finally, if it found a desirable solution, it will write back to
IR.

LoopPlan to LoopPlan transformations and cost models support a wide
range of optimization attempts. It also an open system. New
transformations can be added at any time.

Oh, wait the inner three loops are a DGEMM, let's do a library call.

VPlan : Hierarchical CFG: A control-flow graph, where nodes are blocks or
hierarchical CFG's.

## PGO

In addition to SCEV, [Block
Frequency](https://llvm.org/docs/BlockFrequencyTerminology.html) can
be used to identify hot resp. cold sub-loops.

## Inlining and Outlining

xxxx

# VPlan

Loop vectorization is a kind of loop optimization. Should there be in
the end only one loop optimization pass that optimizes the loop nest
including vectorization?

# Design Principles

New passes are fat, do abstraction raising, perform state to state
transforms, cost modeling, and only write changes back to IR.

Wegman and Zadeck presented an algorithm that combines *constant
propagation* and *unreachable-code elimination*: IPSCCP in LLVM and
MLIR.  Click and Cooper talked about fat passes, i.e., *constant
propagation*, *value numbering*, and *uncreachable code
elimination*. They showed that fat passes can be more powerful.

# Open Questions:

* Which data structure to represent the LoopPlan?
* Which optmizations are performed on the LoopPlanner?
* Merging efforts with VPlan?

# Literature

* [Loop Optimization Framework](https://arxiv.org/pdf/1811.00632.pdf)
* [A Proposal for A Framework for More Effective Loop Optimization](https://llvm.org/devmtg/2020-09/slides/KruseFinkel-Proposal_for_A_Framework_for_More_Effective_Loop_Optimizations.pdf)
* [Loop Optimizations in LLVM: The Good, The Bad, and The Ugly](https://llvm.org/devmtg/2018-10/slides/Kruse-LoopTransforms.pdf)
* [VPlan](https://llvm.org/docs/Proposals/VectorizationPlan.html)
* [Automatic selection of high-order transformations in the IBM XL FORTRAN compilers](https://www.cs.rice.edu/~vs3/PDF/ibmjrd97.pdf)
* [Lecture Notes on Loop Optimizations](https://www.cs.cmu.edu/~fp/courses/15411-f13/lectures/17-loopopt.pdf)
