#include "TypedSelf.h"

#include "AST/TypedSelf.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Type.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::SelfParam>>
tryParseTypedSelf(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  TypedSelf typedSelf = {tokens.front().getLocation()};

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_MUT) {
    view = view.subspan(1);
    typedSelf.setMut();
  }

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_SELFVALUE) {
    view = view.subspan(1);
    if (view.front().getKind() == TokenKind::Colon) {
      view = view.subspan(1);
      std::optional<std::shared_ptr<ast::types::Type>> type =
          tryParseType(view);
      if (type) {
        typedSelf.setType(*type);
        return std::static_pointer_cast<ast::SelfParam>(
            std::make_shared<TypedSelf>(typedSelf));
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
