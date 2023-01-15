#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/RestPattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "PatternNoTopAlt.h"

#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::tryParseTuplePattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  // FIXME

  assert(false);

  return std::nullopt;
}

} // namespace rust_compiler::parser
