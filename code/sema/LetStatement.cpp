#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeLetStatement(std::shared_ptr<ast::LetStatement> let) {
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern = let->getPattern();

  switch (pattern->getKind()) {
  case ast::patterns::PatternNoTopAltKind::PatternWithoutRange: {
    std::shared_ptr<PatternWithoutRange> woRange =
        std::static_pointer_cast<ast::patterns::PatternWithoutRange>(pattern);
    switch(woRange->getWithoutRangeKind()) {
    case PatternWithoutRangeKind::LiteralPattern: {
      break;
    }
    case PatternWithoutRangeKind::IdentifierPattern: {
      break;
    }
    case PatternWithoutRangeKind::WildcardPattern: {
      break;
    }
    case PatternWithoutRangeKind::RestPattern: {
      break;
    }
    case PatternWithoutRangeKind::ReferencePattern: {
      break;
    }
    case PatternWithoutRangeKind::StructPattern: {
      break;
    }
    case PatternWithoutRangeKind::TupleStructPattern: {
      break;
    }
    case PatternWithoutRangeKind::TuplePattern: {
      break;
    }
    case PatternWithoutRangeKind::GroupedPattern: {
      break;
    }
    case PatternWithoutRangeKind::SlicePattern: {
      break;
    }
    case PatternWithoutRangeKind::PathPattern: {
      break;
    }
    case PatternWithoutRangeKind::MacroInvocation: {
      break;
    }
    }
    break;
  }
  case ast::patterns::PatternNoTopAltKind::RangePattern: {
    break;
  }
  };
}

} // namespace rust_compiler::sema
