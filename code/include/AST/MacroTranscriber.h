#pragma once

#include "AST/AST.h"
#include "AST/DelimTokenTree.h"

#include <memory>

namespace rust_compiler::ast {

class MacroTranscriber : public Node {
  std::shared_ptr<DelimTokenTree> tree;

public:
  MacroTranscriber(Location loc) : Node(loc) {}

  void setTree(std::shared_ptr<DelimTokenTree> t) { tree = t; }
};

} // namespace rust_compiler::ast
