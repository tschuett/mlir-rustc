#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::TypeExpression>>
Parser::tryParseTypePath(std::span<lexer::Token>) {}

std::optional<std::shared_ptr<ast::types::TypeExpression>>
Parser::tryParseParenthesizedType(std::span<lexer::Token>) {}

std::optional<std::shared_ptr<ast::types::TypeExpression>>
Parser::tryParseTypeNoBounds(std::span<lexer::Token>) {}

std::optional<std::shared_ptr<ast::types::TypeExpression>>
Parser::tryParseTypeExpression(std::span<lexer::Token>) {}

} // namespace rust_compiler::parser
