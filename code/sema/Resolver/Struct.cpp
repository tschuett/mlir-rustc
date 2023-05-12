#include "AST/Struct.h"

#include "ADT/CanonicalPath.h"
#include "AST/TupleFields.h"
#include "Basic/Ids.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveStructItem(std::shared_ptr<ast::Struct> str,
                                 const adt::CanonicalPath &prefix,
                                 const adt::CanonicalPath &canonicalPrefix) {
  switch (str->getKind()) {
  case StructKind::StructStruct2:
    return resolveStructStructItem(std::static_pointer_cast<StructStruct>(str),
                                   prefix, canonicalPrefix);
  case StructKind::TupleStruct2:
    return resolveTupleStructItem(std::static_pointer_cast<TupleStruct>(str),
                                  prefix, canonicalPrefix);
  }
}

void Resolver::resolveStructStructItem(
    std::shared_ptr<ast::StructStruct> str, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(str->getNodeId(), str->getIdentifier());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  tyCtx->insertCanonicalPath(str->getNodeId(), cpath);

  // FIXME: experiment
  getTypeScope().insert(segment, str->getNodeId(), str->getLocation(),
                        RibKind::Type);

  resolveVisibility(str->getVisibility());

  NodeId scopeNodeId = str->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (str->hasGenerics())
    resolveGenericParams(str->getGenericParams(), prefix, canonicalPrefix);

  if (str->hasWhereClause())
    resolveWhereClause(str->getWhereClause());

  if (str->hasStructFields()) {
    std::vector<StructField> fields = str->getFields().getFields();
    for (StructField &field : fields) {

      resolveVisibility(field.getVisibility());

      resolveType(field.getType(), prefix, canonicalPrefix);
    }
  }
  getTypeScope().pop();
}

void Resolver::resolveTupleStructItem(
    std::shared_ptr<ast::TupleStruct> tuple, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(tuple->getNodeId(), tuple->getName());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  tyCtx->insertCanonicalPath(tuple->getNodeId(), cpath);

  // FIXME: experiment
  getTypeScope().insert(segment, tuple->getNodeId(), tuple->getLocation(),
                        RibKind::Type);

  resolveVisibility(tuple->getVisibility());

  NodeId scopeNodeId = tuple->getNodeId();
  getTypeScope().push(scopeNodeId);

  if (tuple->hasGenerics())
    resolveGenericParams(tuple->getGenericParams(), prefix, canonicalPrefix);

  if (tuple->hasWhereClause())
    resolveWhereClause(tuple->getWhereClause());

  if (tuple->hasTupleFields()) {
    std::vector<TupleField> fields = tuple->getTupleFields().getFields();
    for (TupleField &field : fields) {

      resolveVisibility(field.getVisibility());

      resolveType(field.getType(), prefix, canonicalPrefix);
    }
  }

  getTypeScope().pop();
}

} // namespace rust_compiler::sema::resolver
