#include "AST/PathIdentSegment.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveType(std::shared_ptr<ast::types::TypeExpression> type) {
  switch (type->getKind()) {
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TypeNoBounds: {
    assert(false && "to be handled later");
    resolveTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(type));
    break;
  }
  }
}

void Resolver::resolveTypeNoBounds(
    std::shared_ptr<ast::types::TypeNoBounds> noBounds) {
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::TypePath: {
    resolveRelativeTypePath(std::static_pointer_cast<TypePath>(noBounds));
    break;
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::RawPointerType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ReferenceType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ArrayType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::SliceType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

/// Note that there is no leading ::
void Resolver::resolveRelativeTypePath(
    std::shared_ptr<ast::types::TypePath> typePath) {
  assert(false && "to be handled later");

  NodeId moduleScopeId = peekCurrentModuleScope();
  NodeId previousResolveNodeId = moduleScopeId;

  std::vector<TypePathSegment> segments = typePath->getSegments();

  assert(segments.size() == 1 && "to be handled later");

  for (unsigned i = 0; i < segments.size(); ++i) {
    TypePathSegment &segment = segments[i];
    PathIdentSegment ident = segment.getSegment();

    NodeId crateScopeId = peekCrateModuleScope();

    if (segment.hasGenerics())
      resolveGenericArgs(segment.getGenericArgs());

    if (segment.hasTypeFunction())
      resolveTypePathFunction(segment.getTypePathFn());

    switch (ident.getKind()) {
    case PathIdentSegmentKind::Identifier: {

      assert(false && "to be handled later");
    }
    case PathIdentSegmentKind::super: {
      assert(false && "to be handled later");
    }
    case PathIdentSegmentKind::self: {
      assert(false && "to be handled later");
    }
    case PathIdentSegmentKind::Self: {
      assert(false && "to be handled later");
    }
    case PathIdentSegmentKind::crate: {
      assert(false && "to be handled later");
    }
    case PathIdentSegmentKind::dollarCrate: {
      assert(false && "to be handled later");
    }
    }
  }
}

void Resolver::resolveTypePathFunction(const ast::types::TypePathFn &) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
