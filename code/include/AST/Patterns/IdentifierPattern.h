#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <string>

namespace rust_compiler::ast::patterns {

class IdentifierPattern : public PatternWithoutRange {
  bool ref = false;
  bool mut = false;
  std::string identifier = "";

public:
  IdentifierPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::IdentifierPattern) {}
  void setRef() { ref = true; }
  void setMut() { mut = true; }
  void setIdentifier(std::string_view id) { identifier = id; }

  std::string getIdentifier();

  std::vector<std::string> getLiterals() override;
};

} // namespace rust_compiler::ast::patterns
