#pragma once

#include <mlir/IR/Operation.h>

namespace rust_compiler::analysis {

class MemorySSA;
class Node;

class MemorySSAWalker {
public:
  MemorySSAWalker(MemorySSA *);

  Node *getClobberingMemoryAccess(const mlir::Operation *);

//  virtual MemoryAccess *getClobberingMemoryAccess(MemoryAccess *,
//                                                  MemoryLocation &) = 0;

private:
  MemorySSA *MSSA;
};

} // namespace rust_compiler::analysis
