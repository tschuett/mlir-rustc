#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::Pattern>>
Parser::tryParsePattern(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> Parser::
    parsePathInExpressionOrStructExprStructOrStructExprUnitOrMacroInvocation() {
  /*
    PathInExpression as PathExpression
    PathInExpression as StructExprStruct
    PathInExpression as StructExprUnit
    SimplePath as MacroInvocation
   */
  xxx
}

} // namespace rust_compiler::parser
