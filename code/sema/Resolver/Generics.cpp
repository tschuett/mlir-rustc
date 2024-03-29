#include "ADT/CanonicalPath.h"
#include "AST/ConstParam.h"
#include "AST/GenericArg.h"
#include "AST/GenericArgs.h"
#include "AST/GenericArgsBinding.h"
#include "AST/GenericArgsConst.h"
#include "AST/GenericParam.h"
#include "AST/TypeBoundWhereClauseItem.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/WhereClauseItem.h"
#include "Resolver.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveWhereClause(const WhereClause &wo,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix) {
  for (auto &item : wo.getItems()) {
    switch (item->getKind()) {
    case WhereClauseItemKind::LifetimeWhereClauseItem: {
      break;
    }
    case WhereClauseItemKind::TypeBoundWherClauseItem: {
      std::shared_ptr<TypeBoundWhereClauseItem> typeBound =
          std::static_pointer_cast<TypeBoundWhereClauseItem>(item);
      resolveType(typeBound->getType(), prefix, canonicalPrefix);
      // FIXME
      if (typeBound->hasTypeParamBounds())
        for (auto &bound : typeBound->getBounds().getBounds())
          resolveTypeParamBound(bound, prefix, canonicalPrefix);
      break;
    }
    }
  }
}

void Resolver::resolveGenericParams(const GenericParams &gp,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix) {
  std::vector<GenericParam> params = gp.getGenericParams();
  for (const GenericParam &par : params) {
    switch (par.getKind()) {
    case GenericParamKind::LifetimeParam: {
      break;
    }
    case GenericParamKind::TypeParam: {
      resolveTypeParam(par, prefix, canonicalPrefix);
      break;
    }
    case GenericParamKind::ConstParam: {
      resolveConstParam(par, prefix, canonicalPrefix);
      break;
    }
    }
  }
}

void Resolver::resolveConstParam(const GenericParam &p,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  ConstParam co = p.getConstParam();

  resolveType(co.getType(), prefix, canonicalPrefix);

  if (co.hasBlock())
    resolveExpression(co.getBlock(), prefix, canonicalPrefix);
  else if (co.hasLiteral())
    resolveExpression(co.getLiteral(), prefix, canonicalPrefix);

  CanonicalPath segment =
      CanonicalPath::newSegment(co.getNodeId(), co.getIdentifier());
  getTypeScope().insert(segment, co.getNodeId(), co.getLocation(),
                        RibKind::Type);

  tyCtx->insertCanonicalPath(co.getNodeId(), segment);
}

void Resolver::resolveTypeParam(const GenericParam &p,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  TypeParam pa = p.getTypeParam();

  if (pa.hasType())
    resolveType(pa.getType(), prefix, canonicalPrefix);

  if (pa.hasTypeParamBounds()) {
    std::vector<std::shared_ptr<ast::types::TypeParamBound>> bounds =
        pa.getBounds().getBounds();
    for (std::shared_ptr<ast::types::TypeParamBound> bound : bounds)
      resolveTypeParamBound(bound, prefix, canonicalPrefix);
  }

  CanonicalPath segment =
      CanonicalPath::newSegment(pa.getNodeId(), pa.getIdentifier());
  getTypeScope().insert(segment, pa.getNodeId(), pa.getLocation(),
                        RibKind::Type);

  tyCtx->insertCanonicalPath(pa.getNodeId(), segment);
}

void Resolver::resolveGenericArgs(const ast::GenericArgs &ga,
                                  const adt::CanonicalPath &prefix,
                                  const adt::CanonicalPath &canonicalPrefix) {
  std::vector<GenericArg> args = ga.getArgs();

  for (GenericArg &a : args) {
    switch (a.getKind()) {
    case GenericArgKind::Lifetime: {
      break;
    }
    case GenericArgKind::Type: {
      resolveType(a.getType(), prefix, canonicalPrefix);
      break;
    }
    case GenericArgKind::Const: {
      GenericArgsConst cons = a.getConst();
      switch (cons.getKind()) {
      case GenericArgsConstKind::BlockExpression: {
        resolveExpression(cons.getBlock(), prefix, canonicalPrefix);
        break;
      }
      case GenericArgsConstKind::LiteralExpression: {
        resolveExpression(cons.getLiteral(), prefix, canonicalPrefix);
        break;
      }
      case GenericArgsConstKind::SimplePathSegment: {
        break;
      }
      }
      break;
    }
    case GenericArgKind::Binding: {
      // FIXME Identifier?
      GenericArgsBinding binding = a.getBinding();
      resolveType(binding.getType(), prefix, canonicalPrefix);
      break;
    }
    }
  }
}

} // namespace rust_compiler::sema::resolver
