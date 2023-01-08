#pragma once

#include "AST/AST.h"
#include "AST/PatternWithoutRange.h"

#include <string>

namespace rust_compiler::ast {

class IdentifierPattern : public PatternWithoutRange {
  bool ref;
  bool mut;
  std::string identifier;

public:
  IdentifierPattern(Location loc) : PatternWithoutRange(loc) {}
  void setRef() { ref = true; }
  void setMut() { mut = true; }
  void setIdentifier(std::string_view id) { identifier = id; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
