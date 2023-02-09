#include "AST/Types/PrimitiveTypes.h"

namespace rust_compiler::ast::types {

//  std::optional<std::string> PrimitiveType2String(PrimitiveTypeKind) {}
//
//  std::optional<PrimitiveTypeKind> isPrimitiveType(std::string_view
//  identifier) {}

size_t PrimitiveType::getTokens() { return 1; }

} // namespace rust_compiler::ast::types
