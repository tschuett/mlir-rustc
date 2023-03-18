#include "ADT/CanonicalPath.h"
#include "Resolver.h"

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::resolver {

void Resolver::resolveModule(std::shared_ptr<ast::Module>,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveStaticItem(std::shared_ptr<ast::StaticItem> stat,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(stat->getNodeId(), stat->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(stat->getNodeId(), cpath);

  resolveType(stat->getType());
  if (stat->hasInit())
    resolveExpression(stat->getInit(), path, cpath);
}

void Resolver::resolveConstantItem(std::shared_ptr<ast::ConstantItem> cons,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(cons->getNodeId(), cons->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(cons->getNodeId(), cpath);

  resolveVisibility(cons->getVisibility());

  resolveType(cons->getType());
  if (cons->hasInit())
    resolveExpression(cons->getInit(), path, cpath);
}

} // namespace rust_compiler::sema::resolver
