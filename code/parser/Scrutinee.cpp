#include "Parser/Parser.h"

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Scrutinee>>
Parser::tryParseScrutinee(std::span<lexer::Token> tokens) {}

} // namespace rust_compiler::ast
