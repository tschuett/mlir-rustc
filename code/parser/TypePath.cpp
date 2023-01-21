#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
Parser::tryParseTypePath(std::span<lexer::Token> tokens) {}

} // namespace rust_compiler::parser
