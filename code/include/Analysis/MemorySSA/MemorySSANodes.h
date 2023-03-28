#pragma once

#include "mlir/IR/Operation.h"

#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::analysis {

enum class NodeType { Root, Def, Use, Phi, Term };

class MemoryAccess {

public:
  static inline bool classof(const MemoryAccess *) { return true; }

  mlir::Block *getBlock() const { return block; }

protected:
  MemoryAccess(mlir::Block *block) : block(block) {}

private:
  MemoryAccess(const MemoryAccess &);
  void operator=(const MemoryAccess &);

  mlir::Block *block;
};

class MemoryUseOrDef : public MemoryAccess {
public:
  /// \brief Get the instruction that this MemoryUse represents.
  mlir::Operation *getMemoryInst() const { return memoryInst; }

protected:
  MemoryUseOrDef(MemoryAccess *dma, mlir::Operation *op, mlir::Block *block)
      : MemoryAccess(block), memoryInst(op) {
    setDefiningAccess(dma);
  }

  void setDefiningAccess(MemoryAccess *dma);

private:
  mlir::Operation *memoryInst;
};

class MemoryDef final : public MemoryUseOrDef {

public:
  MemoryDef(MemoryAccess *DMA, mlir::Operation *op, mlir::Block *block)
      : MemoryUseOrDef(DMA, op, block) {}
};

class MemoryUse final : public MemoryUseOrDef {

public:
  MemoryUse(MemoryAccess *DMA, mlir::Operation *op, mlir::Block *block)
      : MemoryUseOrDef(DMA, op, block) {}
};

class MemoryPhi final : public MemoryAccess {

public:
  MemoryPhi(mlir::Block *block) : MemoryAccess(block) {}
};

class Node {

public:
  void setDominator(std::shared_ptr<Node>);
  void setPostDominator(std::shared_ptr<Node>);

private:
  Node(mlir::Operation *op, NodeType t,
       llvm::ArrayRef<std::shared_ptr<Node>> a);
  //  Node(mlir::Operation *op, NodeType t, std::shared_ptr<Node> arg);
  Node(mlir::Operation *op, NodeType t,
       std::optional<std::shared_ptr<Node>> arg);
  Node() = default;

  friend class MemorySSA;
};

} // namespace rust_compiler::analysis
