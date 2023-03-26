#include "ADT/CanonicalPath.h"
#include "AST/Expression.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolvePatternDeclaration(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern, RibKind kind) {
  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<std::string>())};

  resolvePatternDeclarationWithBindings(pattern, kind, bindings);
}

void Resolver::resolvePatternDeclarationWithBindings(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> noTopAlt, RibKind ribKind,
    std::vector<PatternBinding> &bindings) {
  switch (noTopAlt->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    resolvePatternDeclarationWithoutRange(
        std::static_pointer_cast<PatternWithoutRange>(noTopAlt), ribKind,
        bindings);
    break;
  }
  case PatternNoTopAltKind::RangePattern: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolvePatternDeclarationWithoutRange(
    std::shared_ptr<ast::patterns::PatternWithoutRange> pat, RibKind rib,
    std::vector<PatternBinding> &bindings) {
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
    resolvePathPatternDeclaration(std::static_pointer_cast<PathPattern>(pat),
                                  rib);
    break;
  }
  case PatternWithoutRangeKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolvePathPatternDeclaration(
    std::shared_ptr<ast::patterns::PathPattern> pat, RibKind rib) {
  std::shared_ptr<ast::Expression> path = pat->getPath();

  assert(false && "to be handled later");

  if (path->getExpressionKind() ==
      ast::ExpressionKind::ExpressionWithoutBlock) {
    std::shared_ptr<ast::ExpressionWithoutBlock> pathWoBlock =
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(path);
    if (pathWoBlock->getWithoutBlockKind() ==
        ast::ExpressionWithoutBlockKind::PathExpression) {
      std::shared_ptr<ast::PathExpression> pathExpr =
          std::static_pointer_cast<ast::PathExpression>(pathWoBlock);
      if (pathExpr->getPathExpressionKind() ==
          ast::PathExpressionKind::PathInExpression) {
        resolvePathExpression(
            std::static_pointer_cast<ast::PathInExpression>(pathExpr));
      }
    }
  }
}

} // namespace rust_compiler::sema::resolver
