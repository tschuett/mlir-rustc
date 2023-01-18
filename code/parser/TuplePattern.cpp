#include "AST/Patterns/TuplePattern.h"

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/RestPattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"
#include "PatternNoTopAlt.h"

#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::tryParseTuplePattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  if (view.front().getKind() == TokenKind::ParenOpen) {
    view = view.subspan(1);

    std::optional<std::shared_ptr<ast::patterns::TuplePatternItems>> items =
        tryParseTuplePatternItems(view);
    if (items) {
      view = view.subspan((*items)->getTokens());
      TuplePattern tuple = {tokens.front().getLocation()};
      tuple.add(*items);
      return std::static_pointer_cast<PatternNoTopAlt>(
          std::make_shared<TuplePattern>(tuple));
    }
  }
  // FIXME

  assert(false);

  return std::nullopt;
}

} // namespace rust_compiler::parser
