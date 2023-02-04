#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::analysis {

enum class NodeType { Root, Def, Use, Phi, Term };

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
