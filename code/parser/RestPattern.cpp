#include "AST/Patterns/RestPattern.h"

#include "AST/Patterns/PatternNoTopAlt.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "PatternNoTopAlt.h"

#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::tryParseRestPattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == lexer::TokenKind::DotDot) {
    return std::static_pointer_cast<patterns::PatternNoTopAlt>(
        std::make_shared<patterns::RestPattern>(view.front().getLocation()));
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
