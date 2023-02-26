#pragma once

#include "AST/AST.h"
#include "AST/MacroMatcher.h"
#include "AST/MacroTranscriber.h"

namespace rust_compiler::ast {

class MacroRule : public Node {
  MacroMatcher matcher;
  MacroTranscriber transcriber;

public:
  MacroRule(Location loc) : Node(loc), matcher(loc), transcriber(loc) {}

  void setMatcher(const MacroMatcher &m) { matcher = m; }
  void setTranscriber(const MacroTranscriber &m) { transcriber = m; }
};

} // namespace rust_compiler::ast
