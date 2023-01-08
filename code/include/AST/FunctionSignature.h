#pragma once

#include "AST/AST.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/GenericParams.h"
#include "AST/Types/Types.h"
#include "AST/WhereClause.h"
#include "Location.h"

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class FunctionSignature {
  // std::vector<Argument> args;
  std::string name;
  std::shared_ptr<ast::types::Type> returnType;
  FunctionQualifiers qual;
  FunctionParameters parm;
  WhereClause where;
  std::shared_ptr<GenericParams> gen;

public:
  FunctionSignature(Location loc) : qual(loc), where(loc){};

  std::string getName();

  void setName(std::string_view name);

  void setQualifiers(FunctionQualifiers qual);

  void setGenericParams(std::shared_ptr<GenericParams> gen);

  void setParameters(FunctionParameters parm);

  FunctionParameters getParameters() const { return parm; }

  void setReturnType(std::shared_ptr<ast::types::Type> returnType);

  std::shared_ptr<ast::types::Type> getReturnType() const { return returnType; }

  void setWhereClause(WhereClause where);
};

} // namespace rust_compiler::ast

// TODO extern Abi
