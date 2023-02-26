#pragma once

#include "AST/AST.h"
#include "AST/DelimTokenTree.h"

namespace rust_compiler::ast {

class MacroTranscriber : public Node {
  DelimTokenTree tree;

public:
 MacroTranscriber(Location loc) : Node(loc), tree(loc) {}

  void setTree(const DelimTokenTree &t) { tree = t; }
};

} // namespace rust_compiler::ast
