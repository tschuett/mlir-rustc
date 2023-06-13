#include "AST/ConstantItem.h"
#include "AST/Module.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/ImplTraitType.h"
#include "AST/Types/TraitObjectType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;

namespace rust_compiler::sema {

void Sema::walkType(ast::types::TypeExpression *type) {
  switch (type->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    assert(false);
    walkTypeNoBounds(static_cast<ast::types::TypeNoBounds *>(type));
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false);
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false);
  }
  }
}

void Sema::walkTypeNoBounds(ast::types::TypeNoBounds *noBounds) {
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false);
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false);
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false);
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    assert(false);
  }
  case TypeNoBoundsKind::TypePath: {
    assert(false);
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false);
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false);
  }
  case TypeNoBoundsKind::RawPointerType: {
    assert(false);
  }
  case TypeNoBoundsKind::ReferenceType: {
    assert(false);
  }
  case TypeNoBoundsKind::ArrayType: {
    ast::types::ArrayType *array =
        static_cast<ast::types::ArrayType *>(noBounds);
    [[maybe_unused]] bool isConst =
        isConstantExpression(array->getExpression().get());
    assert(false);
  }
  case TypeNoBoundsKind::SliceType: {
    assert(false);
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false);
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false);
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false);
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false);
  }
  }
}

std::pair<size_t, size_t>
Sema::getAlignmentAndSizeOfType(ast::types::TypeExpression *type) {
  switch (type->getKind()) {
  case TypeExpressionKind::TypeNoBounds:
    return getAlignmentAndSizeOfTypeNoBounds(
        static_cast<ast::types::TypeNoBounds *>(type));
  case TypeExpressionKind::ImplTraitType:
    return getAlignmentAndSizeOfImplTraitType(
        static_cast<ast::types::ImplTraitType *>(type));
  case TypeExpressionKind::TraitObjectType:
    return getAlignmentAndSizeOfTraitObjectType(
        static_cast<ast::types::TraitObjectType *>(type));
  }
}

std::pair<size_t, size_t>
Sema::getAlignmentAndSizeOfTypeNoBounds(ast::types::TypeNoBounds *) {
  assert(false);
}

std::pair<size_t, size_t>
Sema::getAlignmentAndSizeOfImplTraitType(ast::types::ImplTraitType *) {
  assert(false);
}

std::pair<size_t, size_t>
Sema::getAlignmentAndSizeOfTraitObjectType(ast::types::TraitObjectType *) {
  assert(false);
}

} // namespace rust_compiler::sema
