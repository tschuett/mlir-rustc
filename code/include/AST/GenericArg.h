#pragma once

#include "AST/AST.h"
#include "AST/GenericArgsBinding.h"
#include "AST/GenericArgsConst.h"
#include "AST/Lifetime.h"
#include "AST/Types/TypeExpression.h"

#include <optional>

namespace rust_compiler::ast {

enum class GenericArgKind { Lifetime, Type, Const, Binding };

class GenericArg : public Node {
  std::optional<Lifetime> lifetime;
  std::shared_ptr<ast::types::TypeExpression> type;
  std::optional<GenericArgsConst> argsConst;
  std::optional<GenericArgsBinding> binding;

  GenericArgKind kind;

public:
  GenericArg(Location loc) : Node(loc){};

  void setLifetime(const Lifetime &l) {
    lifetime = l;
    kind = GenericArgKind::Lifetime;
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> t) {
    type = t;
    kind = GenericArgKind::Type;
  }
  void setArgsConst(const GenericArgsConst &a) {
    argsConst = a;
    kind = GenericArgKind::Const;
  }
  void setArgsBinding(const GenericArgsBinding &b) {
    binding = b;
    kind = GenericArgKind::Binding;
  }

  GenericArgKind getKind() const { return kind; }

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }
  GenericArgsBinding getBinding() const { return *binding; }
  GenericArgsConst getConst() const { return *argsConst; }
};

} // namespace rust_compiler::ast
