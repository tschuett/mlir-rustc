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

class FunctionSignature : public Node {
  // std::vector<Argument> args;
  std::string name;
  std::shared_ptr<ast::types::Type> returnType;
  FunctionQualifiers qual;
  FunctionParameters parm;
  std::shared_ptr<WhereClause> where;
  std::shared_ptr<GenericParams> gen;

public:
  FunctionSignature(Location loc)
      : Node(loc), qual(loc), parm(loc){};

  std::string getName();

  void setName(std::string_view name);

  void setQualifiers(FunctionQualifiers qual);

  void setGenericParams(std::shared_ptr<GenericParams> gen);

  void setParameters(FunctionParameters parm);

  FunctionParameters getParameters() const { return parm; }

  void setReturnType(std::shared_ptr<ast::types::Type> returnType);

  std::shared_ptr<ast::types::Type> getReturnType() const { return returnType; }

  void setWhereClause(std::shared_ptr<WhereClause> where);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast

// TODO extern Abi
