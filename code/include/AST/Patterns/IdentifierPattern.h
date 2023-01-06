#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <string>

namespace rust_compiler::ast::patterns {

class IdentifierPattern : public PatternWithoutRange {
  bool ref;
  bool mut;
  std::string identifier;

public:
  void setRef() { ref = true; }
  void setMut() { mut = true; }
  void setIdentifier(std::string_view id) { identifier = id; }
};

} // namespace rust_compiler::ast::patterns
