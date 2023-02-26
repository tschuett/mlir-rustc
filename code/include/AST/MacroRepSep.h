#pragma once

#include "AST/AST.h"
#include "Lexer/Token.h"

#include <optional>

namespace rust_compiler::ast {

class MacroRepSep : public Node {
  std::optional<lexer::Token> token;

public:
 MacroRepSep(Location loc) : Node(loc) {}

  void setToken(lexer::Token t) { token = t; }
};

} // namespace rust_compiler::ast
