#include "ShorthandSelf.h"

#include "AST/SelfParam.h"
#include "AST/ShorthandSelf.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::SelfParam>>
tryParseShorthandSelf(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  ShorthandSelf shortSelf = {tokens.front().getLocation()};

  if (view.front().getKind() == TokenKind::And) {
    view = view.subspan(1);
    shortSelf.setAnd();
  }

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_MUT) {
    view = view.subspan(1);
    shortSelf.setMut();
  }

  return std::static_pointer_cast<ast::SelfParam>(
      std::make_shared<ShorthandSelf>(shortSelf));
}

} // namespace rust_compiler::parser


// FIXME lifetime
