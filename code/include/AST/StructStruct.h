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
};

} // namespace rust_compiler::ast
