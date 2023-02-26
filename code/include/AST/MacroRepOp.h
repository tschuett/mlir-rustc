#pragma once

#include "AST/AST.h"

namespace rust_compiler::ast {

enum class MacroRepOpKind { Star, Plus, Qmark };

class MacroRepOp : public Node {
  MacroRepOpKind kind;

public:
  MacroRepOp(Location loc) : Node(loc) {}

  void setKind(MacroRepOpKind k) { kind = k; }
};

} // namespace rust_compiler::ast
