#pragma once

#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/GenericParams.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"

namespace rust_compiler::ast {

class BlockExpression;

class Function : public VisItem {
  std::shared_ptr<Expression> body;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Function, vis), qualifiers(loc) {}

  bool hasBody() const;

  std::shared_ptr<Expression> getBody();

  void setQualifiers(FunctionQualifiers qualifiers);

  void setGenericParams(GenericParams genericParams);

  void setWhereClasue(WhereClause whereClause);

  void setParameters(FunctionParameters functionParameters);

  void setBody(std::shared_ptr<Expression> block);

  void setReturnType(std::shared_ptr<ast::types::TypeExpression> returnType);
};

} // namespace rust_compiler::ast

// BlockExpression
