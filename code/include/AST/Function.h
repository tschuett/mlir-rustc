#pragma once

#include "AST/FunctionQualifiers.h"
#include "AST/GenericParams.h"
#include "AST/VisItem.h"
#include "AST/WhereClause.h"
#include "AST/FunctionParameters.h"

namespace rust_compiler::ast {

class BlockExpression;

class Function : public VisItem {
  std::shared_ptr<BlockExpression> body;
  FunctionQualifiers qualifiers;

public:
  Function(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Function, vis), qualifiers(loc) {}

  bool hasBody() const;

  std::shared_ptr<BlockExpression> getBody();

  void setQualifiers(FunctionQualifiers qualifiers);

  void setGenericParams(GenericParams genericParams);

  void setWhereClasue(WhereClause whereClause);

    void setParameters(FunctionParameters functionParameters);

  void setBody(std::shared_ptr<BlockExpression> block);

  void setReturnType(std::shared_ptr<ast::types::TypeExpression> returnType);

};

} // namespace rust_compiler::ast

// BlockExpression
