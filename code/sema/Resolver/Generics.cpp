#include "ADT/CanonicalPath.h"
#include "AST/ConstParam.h"
#include "AST/GenericArg.h"
#include "AST/GenericArgs.h"
#include "AST/GenericArgsBinding.h"
#include "AST/GenericArgsConst.h"
#include "AST/GenericParam.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "Resolver.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveWhereClause(const WhereClause &) {
  // FIXME
  assert(false && "to be handled later");
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
    for (std::shared_ptr<ast::types::TypeParamBound> bound : bounds) {
      resolveTypeParamBound(bound);
    }
  }

  CanonicalPath segment =
      CanonicalPath::newSegment(p.getNodeId(), pa.getIdentifier());
  getTypeScope().insert(segment, pa.getNodeId(), pa.getLocation(),
                        RibKind::Type);

  tyCtx->insertCanonicalPath(pa.getNodeId(), segment);
  assert(false);
}

void Resolver::resolveTypeParamBound(
    std::shared_ptr<ast::types::TypeParamBound> bound) {
  assert(false);
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
      GenericArgsBinding binding = a.getBinding();
      resolveType(binding.getType(), prefix, canonicalPrefix);
      break;
    }
    }
  }
}

} // namespace rust_compiler::sema::resolver
