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
  StructStruct(const adt::CanonicalPath &path, Location loc)
      : Struct(path, loc, StructKind::StructStruct) {}
};

} // namespace rust_compiler::ast