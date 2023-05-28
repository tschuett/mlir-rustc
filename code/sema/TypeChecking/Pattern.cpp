#include "AST/PathExpression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/RangePattern.h"
#include "Lexer/Token.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkPattern(std::shared_ptr<ast::patterns::Pattern> pat,
                           TyTy::BaseType *rhs) {
  for (auto &p : pat->getPatterns()) {
    TyTy::BaseType *infered = checkPattern(p, rhs);
    if (infered->getKind() != TyTy::TypeKind::Error)
      return infered;
  }

  llvm::errs() << "failed to check pattern declaration"
               << "\n";
  return new TyTy::ErrorType(pat->getNodeId());
}

TyTy::BaseType *
TypeResolver::checkPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat,
                           TyTy::BaseType *t) {
  TyTy::BaseType *infered = nullptr;

  switch (pat->getKind()) {
  case PatternNoTopAltKind::PatternWithoutRange: {
    infered = checkPatternWithoutRange(
        std::static_pointer_cast<PatternWithoutRange>(pat), t);
    break;
  }
  case PatternNoTopAltKind::RangePattern: {
    infered = checkRangePattern(std::static_pointer_cast<RangePattern>(pat), t);
    break;
  }
  }

  if (infered == nullptr) {
    llvm::errs() << "failed to check pattern declaration"
                 << "\n";
    return new TyTy::ErrorType(pat->getNodeId());
  }

  tcx->insertType(pat->getIdentity(), infered);

  return infered;
}

TyTy::BaseType *TypeResolver::checkPatternWithoutRange(
    std::shared_ptr<ast::patterns::PatternWithoutRange> pat,
    TyTy::BaseType *ty) {
  switch (pat->getWithoutRangeKind()) {
  case PatternWithoutRangeKind::LiteralPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::IdentifierPattern: {
    return ty;
  }
  case PatternWithoutRangeKind::WildcardPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::RestPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::ReferencePattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::StructPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::TupleStructPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::TuplePattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::GroupedPattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::SlicePattern: {
    assert(false && "to be implemented");
  }
  case PatternWithoutRangeKind::PathPattern: {
    return checkPathPattern(std::static_pointer_cast<PathPattern>(pat), ty);
  }
  case PatternWithoutRangeKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *
TypeResolver::checkRangePattern(std::shared_ptr<ast::patterns::RangePattern> pt,
                                TyTy::BaseType *t) {
  assert(false && "to be implemented");
  switch (pt->getRangeKind()) {
  case RangePatternKind::RangeInclusivePattern: {
    assert(false && "to be implemented");
  }
  case RangePatternKind::RangeFromPattern: {
    assert(false && "to be implemented");
  }
  case RangePatternKind::RangeToInclusivePattern: {
    assert(false && "to be implemented");
  }
  case RangePatternKind::ObsoleteRangePattern: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *
TypeResolver::checkPathPattern(std::shared_ptr<ast::patterns::PathPattern> pat,
                               TyTy::BaseType *) {
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
