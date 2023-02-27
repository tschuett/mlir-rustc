#pragma once

#include "AST/AST.h"
#include "AST/MacroFragSpec.h"
#include "AST/MacroMatcher.h"
#include "AST/MacroRepOp.h"
#include "AST/MacroRepSep.h"
#include "Lexer/Token.h"

namespace rust_compiler::ast {

class MacroMatch : public Node {

  std::optional<lexer::Token> token;

public:
  MacroMatch(Location loc) : Node(loc) {}

  void setToken(const lexer::Token &tok) { token = tok; }
};

} // namespace rust_compiler::ast
