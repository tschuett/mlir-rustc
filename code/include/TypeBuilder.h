#pragma once

#include "AST/Type.h"

#include <memory>
#include <mlir/IR/BuiltinTypes.h>

namespace rust_compiler {

class TypeBuilder {
public:
  mlir::Type getType(std::shared_ptr<ast::Type>);
};

} // namespace rust_compiler
