#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <string>

namespace rust_compiler::ast::patterns {

class IdentifierPattern : public PatternWithoutRange {
  bool ref = false;
  bool mut = false;
  std::string identifier;

public:
  IdentifierPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::IdentifierPattern) {}
  void setRef() { ref = true; }
  void setMut() { mut = true; }
  void setIdentifier(std::string_view id) { identifier = id; }
  size_t getTokens() override;

  std::string getIdentifier();
};

} // namespace rust_compiler::ast::patterns
