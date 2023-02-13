#pragma once

#include "AST/AST.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Visiblity.h"
#include "Location.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class WhereClause;
class GenericParams;

class FunctionSignature : public Node {
  // std::vector<Argument> args;
  std::string name;
  std::shared_ptr<ast::types::TypeExpression> returnType;
  FunctionQualifiers qual;
  FunctionParameters parm;
  std::shared_ptr<WhereClause> where;
  std::shared_ptr<GenericParams> gen;
  std::optional<Visibility> vis;

public:
  FunctionSignature(Location loc) : Node(loc), qual(loc), parm(loc){};

  std::string getName();

  void setName(std::string_view name);

  void setQualifiers(FunctionQualifiers qual);

  void setGenericParams(std::shared_ptr<GenericParams> gen);

  void setParameters(FunctionParameters parm);

  FunctionParameters getParameters() const { return parm; }

  void setReturnType(std::shared_ptr<ast::types::TypeExpression> returnType);

  bool hasReturnType() const;

  std::shared_ptr<ast::types::TypeExpression> getReturnType() const { return returnType; }

  void setWhereClause(std::shared_ptr<WhereClause> where);

  void setVisibility(Visibility _vis) { vis = _vis; }
};

} // namespace rust_compiler::ast

// TODO extern Abi
