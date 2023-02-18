#pragma once

#include "AST/GenericParams.h"
#include "AST/Struct.h"
#include "AST/StructFields.h"
#include "AST/WhereClause.h"

#include <optional>

namespace rust_compiler::ast {

class StructStruct : public Struct {
  std::string identifier;
  std::optional<GenericParams> genericParams;
  std::optional<WhereClause> whereClause;
  std::optional<StructFields> structFields;

public:
  StructStruct(Location loc, std::optional<Visibility> vis)
      : Struct(loc, StructKind::StructStruct, vis) {}

  void setWhereClause(const WhereClause &wc) { whereClause = wc; }
  void setGenericParams(const GenericParams &gp) { genericParams = gp; }
  void setIdentifier(std::string_view id) { identifier = id; }
};

} // namespace rust_compiler::ast
