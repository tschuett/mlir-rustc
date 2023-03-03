#include "Parser/Parser.h"

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseBinaryExpression(bool allowBlocks) {}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseUnaryExpression(bool allowBlocks) {}

} // namespace rust_compiler::parser
