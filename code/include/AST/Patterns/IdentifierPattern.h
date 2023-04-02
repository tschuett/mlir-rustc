#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Lexer/Identifier.h"

#include <memory>
#include <string>

namespace rust_compiler::ast::patterns {

using namespace rust_compiler::lexer;

class IdentifierPattern : public PatternWithoutRange {
  bool ref = false;
  bool mut = false;
  Identifier identifier;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern;

public:
  IdentifierPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::IdentifierPattern) {}
  void setRef() { ref = true; }
  void setMut() { mut = true; }
  void setIdentifier(const Identifier &id) { identifier = id; }

  void addPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat) {
    pattern = pat;
  }

  Identifier getIdentifier() const { return identifier; }
  bool hasMut() const { return mut; }
  bool hasRef() const { return ref; }
};

} // namespace rust_compiler::ast::patterns
