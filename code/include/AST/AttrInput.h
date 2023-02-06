#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Expression.h"

#include <memory>
#include <variant>

namespace rust_compiler::ast {

enum class AttrInputKind { DelimTokenTree, Expression };

class AttrInput {
  std::variant<DelimTokenTree, std::shared_ptr<Expression>> input;

public:
  AttrInputKind getKind() const;
};

} // namespace rust_compiler::ast
