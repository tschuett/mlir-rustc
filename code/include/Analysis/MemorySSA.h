#pragma once

#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <optional>

namespace mlir {
struct LogicalResult;
class Operation;
class Region;
} // namespace mlir

namespace rust_compiler::analysis {

class Node;

enum class NodeType { Root, Def, Use, Phi, Term };

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
};

std::optional<MemorySSA> buildMemorySSA(::mlir::Region &region);

} // namespace rust_compiler::analysis
