#include "Analysis/MemorySSA.h"

#include "llvm/ADT/STLExtras.h"

#include <llvm/ADT/simple_ilist.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::analysis {

struct MemorySSA::Node : public llvm::ilist_node<Node> {
  using Type = MemorySSA::NodeType;

  void setArgument(unsigned i, Node *node) {
    assert(i < argCount);
    if (nullptr != args[i].arg)
      args[i].arg->users.erase(args[i].getIterator());

    args[i].arg = node;
    if (nullptr != node)
      node->users.push_back(args[i]);
  }

private:
  Node() = default;
  Node(const Node &) = delete;
  Node(mlir::Operation *op, Type t, llvm::ArrayRef<Node *> a) {
    assert(a.size() == 1 || t == Type::Phi);
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
  Type type = Type::Root;
  unsigned argCount = 0;

  struct Arg : public llvm::ilist_node<Arg> {
    Node *arg = nullptr;
    unsigned index = 0;

    //    Node *getParent() {
    //      return args[index].arg;
    ////      auto offset =
    ////          static_cast<unsigned>(offsetof(Node, args) + sizeof(Arg) *
    ///index); /      return reinterpret_cast<Node *>(reinterpret_cast<char
    ///*>(this) - offset);
    //    }
  };

  llvm::simple_ilist<Arg> users;

  std::vector<Arg> args; // Variadic size
};

MemorySSA::Node *memSSAProcessRegion(mlir::Region &region,
                                     MemorySSA::Node *entryNode,
                                     MemorySSA &memSSA) {
  assert(nullptr != entryNode);
  // Only structured control flow is supported for now
  // if (!llvm::hasSingleElement(region))

  return nullptr;
}

std::optional<MemorySSA> buildMemorySSA(::mlir::Region &region) {
  MemorySSA ret;
  if (auto last = memSSAProcessRegion(region, ret.getRoot(), ret)) {
    ret.getTerm()->setArgument(0, last);
  } else {
    return std::nullopt;
  }
  return std::move(ret);
}

} // namespace rust_compiler::analysis

// https://discourse.llvm.org/t/upstreaming-from-our-mlir-python-compiler-project/64931/4
