#include "TypeBuilder.h"

#include "AST/Types/PrimitiveTypes.h"
#include "AST/Types/Types.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace rust_compiler {

mlir::Type TypeBuilder::getType(std::shared_ptr<ast::types::Type> type){};

} // namespace rust_compiler
