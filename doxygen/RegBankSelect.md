[GlobalISel] Introduce global variant of regbankselect
* https://reviews.llvm.org/D90304



```cpp
enum RegisterBankKind {
  // R0 - R15
  GPR,
  // D0 - D31
  FPR,
  // R or D registers
  GPROrFPR,
  // Z0 - Z31
  SVEData,
  // P0 - P15
  SVEPredicate
};
```

```cpp
RegisterBankKind classifyDef(const MachineIntr&);

bool isDomainReassignable(const MachineInst%);

bool usesFPR(const MachineInst%);
```


* Algorithm

1. assign unambiguous register banks
2. disambiguate ambiguous register banks
  * reassign domain
  * watch uses or defs
  * optimize load and stores?
  * domain knowledge


G_OR, G_FNEG, G_BITCAST, G_STORE, G_SELECT, or G_LOAD

How to detect SVE predicates?


// FIXME: Should be derived from the scheduling model.
How to access scheduling model?


lib/Target/AArch64/AArch64GenSubtargetInfo.inc


https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/TargetOpcodes.def
https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Target/GenericOpcodes.td

https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AArch64/GISel/AArch64LegalizerInfo.cpp
https://llvm.org/docs/GlobalISel/GenericOpcode.html

1. any order

2. ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
   uses before defs?
   for (MachineBasicBlock *MBB : RPOT) {
     for (MachineInstr* mi: reverse(MBB->instrs())) {
     
     }
   }
