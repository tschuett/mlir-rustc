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

# VPlan

Loop vectorization is a kind of loop optimization. Should there be in
the end only one loop optimization pass that optimizes the loop nest
including vectorization?

# Literature

* [Loop Optimization Framework](https://arxiv.org/pdf/1811.00632.pdf)
* [A Proposal for A Framework for More Effective Loop Optimization](https://llvm.org/devmtg/2020-09/slides/KruseFinkel-Proposal_for_A_Framework_for_More_Effective_Loop_Optimizations.pdf)
* [Loop Optimizations in LLVM: The Good, The Bad, and The Ugly](https://llvm.org/devmtg/2018-10/slides/Kruse-LoopTransforms.pdf)

fat passes:
@article{click1995combining,
  title={Combining analyses, combining optimizations},
  author={Click, Cliff and Cooper, Keith D},
  journal={ACM Transactions on Programming Languages and Systems (TOPLAS)},
  volume={17},
  number={2},
  pages={181--196},
  year={1995},
  publisher={ACM New York, NY, USA}
}

@article{10.1145/103135.103136,
    author = {Wegman, Mark N. and Zadeck, F. Kenneth},
    title = {Constant Propagation with Conditional Branches},
    year = {1991},
    issue_date = {April 1991},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {13},
    number = {2},
    issn = {0164-0925},
    url = {https://doi.org/10.1145/103135.103136},
    doi = {10.1145/103135.103136},
    journal = {ACM Trans. Program. Lang. Syst.},
    month = {apr},
    pages = {181–210},
    numpages = {30},
 }
