#include "AST/PathExpression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/RangePattern.h"
#include "AST/Patterns/StructPatternElements.h"
#include "AST/Patterns/TupleStructItems.h"
#include "Basic/Ids.h"
#include "Lexer/Token.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include <cstdlib>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::ast::patterns;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;

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
    return checkStructPattern(std::static_pointer_cast<StructPattern>(pat), ty);
  }
  case PatternWithoutRangeKind::TupleStructPattern: {
    return checkTupleStructPattern(
        std::static_pointer_cast<TupleStructPattern>(pat), ty);
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

TyTy::BaseType *TypeResolver::checkTupleStructPattern(
    std::shared_ptr<ast::patterns::TupleStructPattern> pattern,
    TyTy::BaseType *tuple) {
  TyTy::BaseType *pathType = checkExpression(pattern->getPath());
  if (pathType->getKind() != TypeKind::ADT) {
    llvm::errs() << "expected tuple/struct pattern: " << pathType->toString()
                 << "\n";
    exit(EXIT_FAILURE);
  }

  TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(pathType);

  assert(adt->getNumberOfVariants() > 0);
  TyTy::VariantDef *variant = adt->getVariant(0);
  if (adt->isEnum()) {
    std::optional<basic::NodeId> variantId =
        tcx->lookupVariantDefinition(pattern->getPath()->getNodeId());
    assert(variantId.has_value());
    bool ok = adt->lookupVariantById(*variantId, &variant);
    assert(ok);
  }
  if (variant->getKind() != TyTy::VariantKind::Tuple) {
    llvm::errs() << "expected tuple struct or tuple variant"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  if (pattern->hasItems()) {
    TupleStructItems items = pattern->getItems();
    for (const std::shared_ptr<ast::patterns::Pattern> &pat :
         items.getPatterns()) {
      size_t i = 0;
      for (const std::shared_ptr<ast::patterns::PatternNoTopAlt> &noTop :
           pat->getPatterns()) {
        switch (noTop->getKind()) {
        case PatternNoTopAltKind::RangePattern: {
          llvm_unreachable("no such thing");
          break;
        }
        case PatternNoTopAltKind::PatternWithoutRange: {
          std::shared_ptr<ast::patterns::PatternWithoutRange> woRange =
              std::static_pointer_cast<ast::patterns::PatternWithoutRange>(
                  noTop);
          TyTy::StructFieldType *field = variant->getFieldAt(i++);
          TyTy::BaseType *fieldType = field->getFieldType();

          tcx->insertType(woRange->getIdentity(), fieldType);
          break;
        }
        }
      }
      // FIXME
    }
  }
  return adt;
}

TyTy::BaseType *TypeResolver::checkStructPattern(
    std::shared_ptr<ast::patterns::StructPattern> pattern,
    TyTy::BaseType *type) {
  TyTy::BaseType *patternType = checkExpression(pattern->getPath());

  if (patternType->getKind() != TypeKind::ADT) {
    llvm::errs() << "expected tuple/struct pattern: " << patternType->toString()
                 << "\n";
    exit(EXIT_FAILURE);
  }

  TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(patternType);
  assert(adt->getNumberOfVariants() > 0);
  TyTy::VariantDef *variant = adt->getVariant(0);
  if (adt->isEnum()) {

    std::optional<NodeId> variantId =
        tcx->lookupVariantDefinition(pattern->getPath()->getNodeId());
    assert(variantId.has_value());
    bool ok = adt->lookupVariantById(*variantId, &variant);
    assert(ok);
  }

  if (variant->getKind() != TyTy::VariantKind::Struct) {
    llvm::errs() << "expected struct variant"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  std::vector<Identifier> namedFields;
  if (pattern->hasElements()) {
    StructPatternElements el = pattern->getElements();
    if (el.hasFields()) {
      StructPatternFields fiels = el.getFields();
      for (const StructPatternField &f : fiels.getFields()) {
        switch (f.getKind()) {
        case StructPatternFieldKind::TupleIndex: {
          break;
        }
        case StructPatternFieldKind::Identifier: {
          TyTy::StructFieldType *field = nullptr;
          if (!variant->lookupField(f.getIdentifier(), &field, nullptr)) {
            llvm::errs() << "variant " << variant->getIdentifier().toString()
                         << "does not have field named " << f.getIdentifier().toString()
                         << "\n";
            break;
          }

          namedFields.push_back(f.getIdentifier());
          TyTy::BaseType *fieldType = field->getFieldType();
          checkPattern(f.getPattern(), fieldType);
          tcx->insertType(f.getIdentity(), fieldType);
          break;
        }
        case StructPatternFieldKind::RefMut: {
          TyTy::StructFieldType *field = nullptr;
          if (!variant->lookupField(f.getIdentifier(), &field, nullptr)) {
            llvm::errs() << "variant " << variant->getIdentifier().toString()
                         << "does not have field named " << f.getIdentifier().toString()
                         << "\n";
            break;
          }

          namedFields.push_back(f.getIdentifier());
          TyTy::BaseType *fieldType = field->getFieldType();
          tcx->insertType(f.getIdentity(), fieldType);
          break;
        }
        }
      }
    }
  }

  if (namedFields.size() != variant->getNumberOfFields()) {
    std::map<Identifier, bool> missingNames;
    for (auto &field : variant->getFields())
      missingNames[field->getName()] = true;

    for (auto &named : namedFields)
      missingNames.erase(named);

    for (auto &name : missingNames) {
      llvm::errs() << name.first.toString()
                   << ": is not mentioned in the pattern"
                   << "\n";
    }
  }

  return adt;
}

} // namespace rust_compiler::sema::type_checking
