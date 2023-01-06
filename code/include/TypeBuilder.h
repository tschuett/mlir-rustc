#pragma once

#include "AST/Types/Types.h"

#include <memory>
#include <mlir/IR/BuiltinTypes.h>

namespace rust_compiler {

class TypeBuilder {
public:
  mlir::Type getType(std::shared_ptr<ast::types::Type>);
};

} // namespace rust_compiler
