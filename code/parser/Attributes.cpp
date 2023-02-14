#include "Parser/Parser.h"

#include <optional>
#include <sstream>
#include <vector>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

llvm::Expected<ast::OuterAttribute> Parser::parseOuterAttribute() {}

llvm::Expected<ast::InnerAttribute> Parser::parseInnerAttribute() {}

llvm::Expected<std::vector<ast::OuterAttribute>>
Parser::parseOuterAttributes() {}

llvm::Expected<std::vector<ast::InnerAttribute>>
Parser::parseInnerAttributes() {}

} // namespace rust_compiler::parser
