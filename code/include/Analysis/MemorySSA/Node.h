#pragma once

#include "Analysis/MemorySSA/NodeType.h"

#include <llvm/ADT/simple_ilist.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <cassert>

namespace rust_compiler::analysis {

class Node : public llvm::ilist_node<Node> {

public:
  void setArgument(unsigned i, Node *node);

  void setDominator(Node *node) { dominator = node; }

  void setPostDominator(Node *node) { postDominator = node; }

private:
  Node() = default;
  Node(const Node &) = delete;
  Node(mlir::Operation *op, NodeType t, llvm::ArrayRef<Node *> a) {
    assert(a.size() == 1 || t == NodeType::Phi);
    operation = op;
    argCount = static_cast<unsigned>(a.size());
    type = t;
    for (auto it : llvm::enumerate(a)) {
      auto i = it.index();
      if (i >= 1)
        new (&args[i]) Arg();

      auto arg = it.value();
      args[i].index = static_cast<unsigned>(i);
      if (nullptr != arg) {
        args[i].arg = arg;
        arg->users.push_back(args[i]);
      }
    }
  }
  ~Node() {
    for (unsigned i = 0; i < argCount; ++i) {
      if (args[i].arg != nullptr)
        args[i].arg->users.erase(args[i].getIterator());

      if (i >= 1)
        args[i].~Arg();
    }
  }

  mlir::Operation *operation = nullptr;
  NodeType type = NodeType::Root;
  unsigned argCount = 0;

  Node *dominator = nullptr;
  Node *postDominator = nullptr;

  struct Arg : public llvm::ilist_node<Arg> {
    Node *arg = nullptr;
    unsigned index = 0;

    //    Node *getParent() {
    //      return args[index].arg;
    ////      auto offset =
    ////          static_cast<unsigned>(offsetof(Node, args) + sizeof(Arg) *
    /// index); /      return reinterpret_cast<Node *>(reinterpret_cast<char
    ///*>(this) - offset);
    //    }
  };

  llvm::simple_ilist<Arg> users;

  std::vector<Arg> args; // Variadic size
};

} // namespace rust_compiler::analysis
