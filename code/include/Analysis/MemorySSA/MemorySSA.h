#pragma once

#include "Analysis/MemorySSA/NodeType.h"
#include "Analysis/MemorySSA/Node.h"
#include "Analysis/MemorySSA/NodesIterator.h"

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/Allocator.h>
#include <optional>

namespace mlir {
struct LogicalResult;
class Operation;
class Region;
} // namespace mlir

namespace rust_compiler::analysis {

class MemorySSA {
public:
  MemorySSA() = default;
  MemorySSA(const MemorySSA &) = delete;
  MemorySSA(MemorySSA &&) = default;

  MemorySSA &operator=(const MemorySSA &) = delete;
  MemorySSA &operator=(MemorySSA &&) = default;

  Node *createDef(mlir::Operation *op, Node *arg);
  Node *createUse(mlir::Operation *op, Node *arg);
  Node *createPhi(mlir::Operation *op, llvm::ArrayRef<Node *> args);

  void eraseNode(Node *node);
  NodeType getNodeType(Node *node) const;
  mlir::Operation *getNodeOperation(Node *node) const;
  Node *getNodeDef(Node *node) const;
  llvm::SmallVector<Node *> getUsers(Node *node);

  Node *getRoot();
  Node *getTerm();
  Node *getNode(mlir::Operation *op) const;

  mlir::LogicalResult optimizeUses(
      llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)> mayAlias);

private:
  Node *root = nullptr;
  Node *term = nullptr;
  llvm::DenseMap<mlir::Operation *, Node *> nodesMap;
  llvm::BumpPtrAllocator allocator;
  llvm::simple_ilist<Node> nodes;

  Node *createNode(mlir::Operation *op, NodeType type,
                   llvm::ArrayRef<Node *> args);
};

std::optional<MemorySSA> buildMemorySSA(::mlir::Region &region);

} // namespace rust_compiler::analysis
