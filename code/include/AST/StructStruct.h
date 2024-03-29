#pragma once

#include "AST/GenericParams.h"
#include "AST/OuterAttribute.h"
#include "AST/Struct.h"
#include "AST/StructFields.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class StructStruct : public Struct {
  Identifier identifier;
  std::optional<GenericParams> genericParams;
  std::optional<WhereClause> whereClause;
  std::optional<StructFields> structFields;

public:
  StructStruct(Location loc, std::optional<Visibility> vis)
      : Struct(loc, StructKind::StructStruct2, vis) {}

  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setName(const Identifier id) { identifier = id; }
  void setFields(const StructFields &sf) { structFields = sf; }

  bool hasGenerics() const { return genericParams.has_value(); }
  bool hasWhereClause() const { return whereClause.has_value(); }
  bool hasStructFields() const { return structFields.has_value(); }

  Identifier getIdentifier() const { return identifier; }
  GenericParams getGenericParams() const { return *genericParams; }
  WhereClause getWhereClause() const { return *whereClause; }
  StructFields getFields() const { return *structFields; }
};

} // namespace rust_compiler::ast
