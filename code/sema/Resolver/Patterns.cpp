#include "ADT/CanonicalPath.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolvePatternDeclaration(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> noTopAlt,
    Rib::RibKind ribKind) {
  assert(false && "to be handled later");
  switch (noTopAlt->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    resolvePatternDeclarationWithoutRange(
        std::static_pointer_cast<PatternWithoutRange>(noTopAlt), ribKind);
    break;
  }
  case PatternNoTopAltKind::RangePattern: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolvePatternDeclarationWithoutRange(
    std::shared_ptr<ast::patterns::PatternWithoutRange> pat, Rib::RibKind rib) {
  switch (pat->getWithoutRangeKind()) {
  case PatternWithoutRangeKind::LiteralPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::IdentifierPattern: {
    std::shared_ptr<ast::patterns::IdentifierPattern> id =
        std::static_pointer_cast<IdentifierPattern>(pat);
    getNameScope().insert(
        CanonicalPath::newSegment(id->getNodeId(), id->getIdentifier()),
        id->getNodeId(), id->getLocation(), rib);
    break;
  }
  case PatternWithoutRangeKind::WildcardPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::RestPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::ReferencePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::StructPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::TupleStructPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::TuplePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::GroupedPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::SlicePattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::PathPattern: {
    assert(false && "to be handled later");
  }
  case PatternWithoutRangeKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

} // namespace rust_compiler::sema::resolver
