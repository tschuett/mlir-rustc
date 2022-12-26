#include "MemorySSANode.h"

namespace rust_compiler::analysis {

void Node::setArgument(unsigned i, Node *node) {
  assert(i < argCount);
  if (nullptr != args[i].arg)
    args[i].arg->users.erase(args[i].getIterator());

  args[i].arg = node;
  if (nullptr != node)
    node->users.push_back(args[i]);
}

} // namespace rust_compiler::analysis
