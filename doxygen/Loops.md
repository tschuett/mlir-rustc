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

GCC and LLVM both use loop passes to optimize loops. They both suffer
from pass-ordering issues. In contrast VPlan and SLP both rely on abstraction
raising, evaluate several plans using cost models, and only write back to
IR if they found a better solution.

# The LoopPlaner

The LoopPlanner follows the design of VPLAN. It reads the IR and
transforms it into a higher abstraction representation. It will try
every kind of loop optimization except for vectorization to optimize
the loop nest.

* Abstraction raising
* Legality checks
* LoopPlan to LoopPlan transformations
* Cost Model
* Analysis

## PGO

In addition to SCEV, [Block
Frequency](https://llvm.org/docs/BlockFrequencyTerminology.html) can
be used to identify hot resp. cold sub-loops.

# VPlan

Loop vectorization is a kind of loop optimization. Should there be in
the end only one loop optimization pass that optimizes the loop nest
including vectorization?

# Literature

* [Loop Optimization Framework](https://arxiv.org/pdf/1811.00632.pdf)
* [A Proposal for A Framework for More Effective Loop Optimization](https://llvm.org/devmtg/2020-09/slides/KruseFinkel-Proposal_for_A_Framework_for_More_Effective_Loop_Optimizations.pdf)
* [Loop Optimizations in LLVM: The Good, The Bad, and The Ugly](https://llvm.org/devmtg/2018-10/slides/Kruse-LoopTransforms.pdf)

