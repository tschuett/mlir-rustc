#include "ADT/CanonicalPath.h"
#include "AST/EnumItem.h"
#include "AST/EnumItemDiscriminant.h"
#include "AST/EnumItems.h"
#include "AST/Enumeration.h"
#include "AST/TupleFields.h"
#include "Resolver.h"

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveModule(std::shared_ptr<ast::Module> mod,
                             const adt::CanonicalPath &prefix,
                             const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  tyCtx->insertModule(mod.get());
}

void Resolver::resolveStaticItem(std::shared_ptr<ast::StaticItem> stat,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath decl =
      CanonicalPath::newSegment(stat->getNodeId(), stat->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(stat->getNodeId(), cpath);

  resolveType(stat->getType(), prefix, canonicalPrefix);
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

  resolveType(cons->getType(), prefix, canonicalPrefix);
  if (cons->hasInit())
    resolveExpression(cons->getInit(), path, cpath);
}

void Resolver::resolveEnumerationItem(
    std::shared_ptr<Enumeration> enu, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  CanonicalPath decl =
      CanonicalPath::newSegment(enu->getNodeId(), enu->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(enu->getNodeId(), cpath);
  tyCtx->insertEnumeration(enu->getNodeId(), enu.get());

  resolveVisibility(enu->getVisibility());

  NodeId scopeNodeId = enu->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (enu->hasGenericParams())
    resolveGenericParams(enu->getGenericParams(), prefix, cpath);

  if (enu->hasWhereClause())
    resolveWhereClause(enu->getWhereClause());

  if (enu->hasEnumItems()) {
    std::vector<std::shared_ptr<EnumItem>> it = enu->getEnumItems().getItems();
    for (const auto &i : it) {
      resolveEnumItem(i, path, cpath);
      tyCtx->insertEnumItem(enu.get(), i.get());
    }
  }

  getTypeScope().pop();
}

void Resolver::resolveEnumItem(std::shared_ptr<ast::EnumItem> enuIt,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {

  switch (enuIt->getKind()) {
  case EnumItemKind::Discriminant: {
    EnumItemDiscriminant dis = enuIt->getDiscriminant();
    CanonicalPath decl =
        CanonicalPath::newSegment(dis.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(dis.getNodeId(), cpath);
    break;
  }
  case EnumItemKind::Struct: {
    EnumItemStruct str = enuIt->getStruct();
    CanonicalPath decl =
        CanonicalPath::newSegment(str.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(str.getNodeId(), cpath);

    if (str.hasFields()) {
      std::vector<StructField> fields = str.getFields().getFields();
      for (const StructField &field : fields)
        resolveType(field.getType(), prefix, canonicalPrefix);
    }
    break;
  }
  case EnumItemKind::Tuple: {
    EnumItemTuple tup = enuIt->getTuple();
    CanonicalPath decl =
        CanonicalPath::newSegment(tup.getNodeId(), enuIt->getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(tup.getNodeId(), cpath);
    if (tup.hasTupleFiels()) {
      std::vector<TupleField> fields = tup.getTupleFields().getFields();
      for (const TupleField &tup : fields)
        resolveType(tup.getType(), prefix, canonicalPrefix);
    }
    break;
  }
  }
}

void Resolver::resolveTraitItem(std::shared_ptr<ast::Trait> trait,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveAssociatedItem(
    const ast::AssociatedItem &, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
