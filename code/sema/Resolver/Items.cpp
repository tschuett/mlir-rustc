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

void Resolver::resolveEnumerationItem(
    std::shared_ptr<Enumeration> enu, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
  CanonicalPath decl =
      CanonicalPath::newSegment(enu->getNodeId(), enu->getName());

  CanonicalPath path = prefix.append(decl);
  CanonicalPath cpath = canonicalPrefix.append(decl);

  tyCtx->insertCanonicalPath(enu->getNodeId(), cpath);

  resolveVisibility(enu->getVisibility());

  NodeId scopeNodeId = enu->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (enu->hasGenericParams())
    resolveGenericParams(enu->getGenericParams(), prefix, cpath);

  if (enu->hasWhereClause())
    resolveWhereClause(enu->getWhereClause());

  if (enu->hasEnumItems()) {
    std::vector<EnumItem> it = enu->getEnumItems().getItems();
    for (const EnumItem &i : it)
      resolveEnumItem(i, path, cpath);
  }

  getTypeScope().pop();
}

void Resolver::resolveEnumItem(const ast::EnumItem &enuIt,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {
  switch (enuIt.getKind()) {
  case EnumItemKind::Discriminant: {
    EnumItemDiscriminant dis = enuIt.getDiscriminant();
    CanonicalPath decl =
        CanonicalPath::newSegment(dis.getNodeId(), enuIt.getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(dis.getNodeId(), cpath);
    break;
  }
  case EnumItemKind::Struct: {
    EnumItemStruct str = enuIt.getStruct();
    CanonicalPath decl =
        CanonicalPath::newSegment(str.getNodeId(), enuIt.getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(str.getNodeId(), cpath);

    if (str.hasFields()) {
      std::vector<StructField> fields = str.getFields().getFields();
      for (const StructField &field : fields)
        resolveType(field.getType());
    }
    break;
  }
  case EnumItemKind::Tuple: {
    EnumItemTuple tup = enuIt.getTuple();
    CanonicalPath decl =
        CanonicalPath::newSegment(tup.getNodeId(), enuIt.getName());
    CanonicalPath path = prefix.append(decl);
    CanonicalPath cpath = canonicalPrefix.append(decl);
    tyCtx->insertCanonicalPath(tup.getNodeId(), cpath);
    if (tup.hasTupleFiels()) {
      std::vector<TupleField> fields = tup.getTupleFields().getFields();
      for (const TupleField& tup: fields)
        resolveType(tup.getType());
    }
    break;
  }
  }
}

} // namespace rust_compiler::sema::resolver
