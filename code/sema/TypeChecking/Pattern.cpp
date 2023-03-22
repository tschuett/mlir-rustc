#include "AST/PathExpression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/RangePattern.h"
#include "TyTy.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast::patterns;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat,
                           TyTy::BaseType *t) {
  assert(false && "to be implemented");

  TyTy::BaseType *infered = nullptr;

  switch (pat->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    infered = checkPatternWithoutRange(
        std::static_pointer_cast<PatternWithoutRange>(pat), t);
  }
  case PatternNoTopAltKind::PatternWithoutRange: {
    infered = checkRangePattern(std::static_pointer_cast<RangePattern>(pat), t);
  }
  }

  if (infered == nullptr)
    return new TyTy::ErrorType(pattern->getNodeId());

  tcx->insertType(pat->getIdentity(), infered);

  return infered;
}

TyTy::BaseType *TypeResolver::checkPatternWithoutRange(
    std::shared_ptr<ast::patterns::PatternWithoutRange> pat,
    TyTy::BaseType *ty) {
  assert(false && "to be implemented");
  switch (pat->getWithouRangeKind()) {
  case PatternWithoutRangeKind::LiteralPattern: {
  }
  case PatternWithoutRangeKind::IdentifierPattern: {
    return ty;
  }
  case PatternWithoutRangeKind::WildcardPattern: {
  }
  case PatternWithoutRangeKind::RestPattern: {
  }
  case PatternWithoutRangeKind::ReferencePattern: {
  }
  case PatternWithoutRangeKind::StructPattern: {
  }
  case PatternWithoutRangeKind::TupleStructPattern: {
  }
  case PatternWithoutRangeKind::TuplePattern: {
  }
  case PatternWithoutRangeKind::GroupedPattern: {
  }
  case PatternWithoutRangeKind::SlicePattern: {
  }
  case PatternWithoutRangeKind::PathPattern: {
    return checkPathPattern(std::static_pointer_cast<PathPattern>(pat));
  }
  case PatternWithoutRangeKind::MacroInvocation: {
  }
  }
}

TyTy::BaseType *
TypeResolver::checkRangePattern(std::shared_ptr<ast::patterns::RangePattern> pt,
                                TyTy::BaseType *t) {
  assert(false && "to be implemented");
  switch (pt->getRangeKind()) {
  case RangePatternKind::InclusiveRangePattern: {
  }
  case RangePatternKind::HalfOpenRangePattern: {
  }
  case RangePatternKind::ObsoleteRangePattern: {
  }
  }
}

TyTy::BaseType *
TypeResolver::checkPathPattern(std::shared_ptr<ast::patterns::PathPattern> pat,
                               TyTy::BaseType *) {
  assert(false && "to be implemented");
  std::shared_ptr<ast::PathExpression> path =
      std::static_pointer_cast<ast::PathExpression>(pat->getPath());
  switch (path->getPathExpressionKind()) {
  case ast::PathExpressionKind::PathInExpression: {
    return checkExpression(path);
  }
  case ast::PathExpressionKind::QualifiedPathInExpression: {
    assert(false && "to be implemented");
  }
  }
}

} // namespace rust_compiler::sema::type_checking
