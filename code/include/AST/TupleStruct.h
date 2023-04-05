#pragma once

#include "AST/GenericParams.h"
#include "AST/Struct.h"
#include "AST/TupleFields.h"
#include "AST/WhereClause.h"
#include "Lexer/Identifier.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class TupleStruct : public Struct {
  Identifier identifier;
  std::optional<GenericParams> genericParms;
  std::optional<TupleFields> tupleFields;
  std::optional<WhereClause> whereClause;

public:
  TupleStruct(Location loc, std::optional<Visibility> vis)
      : Struct(loc, StructKind::TupleStruct2, vis) {}

  void setIdentifier(const Identifier& id) { identifier = id; }
  void setGenericParams(const GenericParams &gp) { genericParms = gp; }
  void setWhereClause(const WhereClause &w) { whereClause = w; }
  void setTupleFields(const TupleFields &tp) { tupleFields = tp; }

  Identifier getName() const { return identifier; }

  bool hasGenerics() const { return genericParms.has_value(); }
  GenericParams getGenericParams() const { return *genericParms; };

  bool hasWhereClause() const { return whereClause.has_value(); }
  WhereClause getWhereClause() const { return *whereClause; }

  bool hasTupleFields() const { return tupleFields.has_value(); }
  TupleFields getTupleFields() const { return *tupleFields; }
};

} // namespace rust_compiler::ast
