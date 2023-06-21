#pragma once

#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionReturnType.h"
#include "AST/GenericParams.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"

#include <memory>
#include <optional>
#include <string>

namespace rust_compiler::ast {

class BlockExpression;

class Function : public VisItem {
  std::optional<std::shared_ptr<Expression>> body;
  FunctionQualifiers qualifiers;
  std::optional<FunctionParameters> functionParameters;
  std::optional<GenericParams> genericParams;
  std::optional<FunctionReturnType> returnType;
  std::optional<WhereClause> whereClause;

  Identifier identifier;

public:
  Function(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Function, vis), qualifiers(loc),
        functionParameters(loc), genericParams(loc), whereClause(loc) {}

  bool hasBody() const { return body.has_value(); }

  std::shared_ptr<Expression> getBody() { return *body; }

  void setQualifiers(FunctionQualifiers qualifiers);

  void setGenericParams(GenericParams genericParams);

  void setWhereClasue(WhereClause whereClause);

  void setParameters(FunctionParameters functionParameters);

  void setBody(std::shared_ptr<Expression> block);

  void setReturnType(const FunctionReturnType &ret) { returnType = ret; }

  void setIdentifier(const Identifier &id) { identifier = id; }

  Identifier getName() const {
    return identifier;
  }

  bool hasReturnType() const { return returnType.has_value(); }

  bool hasGenericParams() const { return genericParams.has_value(); }

  GenericParams getGenericParams() const { return *genericParams; }

  std::shared_ptr<ast::types::TypeExpression> getReturnType() const {
    return returnType->getType();
  }

  bool hasWhereClause() const { return whereClause.has_value(); }

  WhereClause getWhereClause() const { return *whereClause; }

  bool hasParams() const { return functionParameters.has_value(); }
  FunctionParameters getParams() { return *functionParameters; }

  FunctionQualifiers getQualifiers() const { return qualifiers; }
};

} // namespace rust_compiler::ast

// BlockExpression
