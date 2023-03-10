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
#include <string>

namespace rust_compiler::ast {

class BlockExpression;

class Function : public VisItem {
  std::shared_ptr<Expression> body;
  FunctionQualifiers qualifiers;
  FunctionParameters functionParameters;
  GenericParams genericParams;
  std::optional<FunctionReturnType> returnType;
  WhereClause whereClause;

  std::string identifier;

public:
  Function(Location loc, std::optional<Visibility> vis)
      : VisItem(loc, VisItemKind::Function, vis), qualifiers(loc),
        functionParameters(loc), genericParams(loc), whereClause(loc) {}

  bool hasBody() const;

  std::shared_ptr<Expression> getBody();

  void setQualifiers(FunctionQualifiers qualifiers);

  void setGenericParams(GenericParams genericParams);

  void setWhereClasue(WhereClause whereClause);

  void setParameters(FunctionParameters functionParameters);

  void setBody(std::shared_ptr<Expression> block);

  void setReturnType(const FunctionReturnType &ret) { returnType = ret; }

  void setIdentifier(std::string_view id) { identifier = id; }

  std::string_view getName() const { return identifier; }

  bool hasReturnType() const { return returnType.has_value(); }

  bool hasGenericParams() const { return genericParams.getNumberOfParams(); }

  GenericParams getGenericParams() const { return genericParams; }

  std::shared_ptr<ast::types::TypeExpression> getReturnType() const {
    return returnType->getType();
  }

  bool hasWhereClause() const { return whereClause.getSize() > 0; }

  WhereClause getWhereClause() const { return whereClause; }

  std::vector<FunctionParamPattern> getParams() {
    std::vector<FunctionParam> params = functionParameters.getParams();
    std::vector<FunctionParamPattern> patterns;

    for (auto &p : params) {
      if (p.getKind() == FunctionParamKind::Pattern && p.getPattern().hasType())
        patterns.push_back(p.getPattern());
    }

    return patterns;
  }
};

} // namespace rust_compiler::ast

// BlockExpression
