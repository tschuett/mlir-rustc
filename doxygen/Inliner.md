Pre-RFC: The Module in- and outliner

# Inlining Take II

The next inliner will do inlining and outlining at module level in one
pass. Outlining might give more opportunities for inlining.


```c
Huge computesAlot(Huge *huge) {
   if (likely) {
     // less code
   } else { // unlikely
     // a lot of code
   }
}
```

Solution:
1. outline else-branch
2. inline function

## InOutlinerState

1. Functions
  1. CallOps
  2. Blocks
     1. Instructions
     2. Allocas
  3. Controlflow Graph
2. Callgraph
3. 

# PGO



# Design Principles

New passes are fat, do abstraction raising, state to state transforms,
cost modeling, and only write changes back to IR.

Wegman and Zadeck presented an algorithm that combines *constant
propagation* and *unreachable-code elimination*. IPSCCP in LLVM.
Click and Cooper talked about fat passes, i.e., *constant
propagation*, *value numbering*, and *uncreachable code elimination*.

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


https://discourse.llvm.org/t/value-range-analysis-of-source-code-variables/62853
