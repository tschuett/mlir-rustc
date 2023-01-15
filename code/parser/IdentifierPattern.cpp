#include "AST/PatternWithoutRange.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "Lexer/KeyWords.h"

#include "Parser/Parser.h"

#include <memory>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::patterns::PatternWithoutRange>>
Parser::tryParseIdentifierPattern(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  IdentifierPattern pattern = {tokens.front().getLocation()};

  if (view.front().isKeyWord()) {
    if (view.front().getKeyWordKind() == KeyWordKind::KW_REF) {
      pattern.setRef();
    }
    if (view.front().getKeyWordKind() == KeyWordKind::KW_MUT) {
      pattern.setMut();
    }
  }

  if (view[1].isKeyWord()) {
    if (view[1].getKeyWordKind() == KeyWordKind::KW_REF) {
      pattern.setRef();
    }
    if (view[1].getKeyWordKind() == KeyWordKind::KW_MUT) {
      pattern.setMut();
    }
  }

  if (view.front().isIdentifier()) {
    pattern.setIdentifier(view.front().getIdentifier());
    return std::static_pointer_cast<PatternWithoutRange>(
        std::make_shared<ast::patterns::IdentifierPattern>(pattern));
  }

  if (view[1].isIdentifier()) {
    pattern.setIdentifier(view[1].getIdentifier());
    return std::static_pointer_cast<PatternWithoutRange>(
        std::make_shared<ast::patterns::IdentifierPattern>(pattern));
  }

  if (view[2].isIdentifier()) {
    pattern.setIdentifier(view[2].getIdentifier());
    return std::static_pointer_cast<PatternWithoutRange>(
        std::make_shared<ast::patterns::IdentifierPattern>(pattern));
  }

  // FIXME add PatternNoTopAlt
  return std::nullopt;
}

} // namespace rust_compiler::parser
