#pragma once

#include "AST/AST.h"
#include "Lexer/Identifier.h"
#include "Lexer/KeyWords.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class SimplePathSegment : public Node {
  std::variant<Identifier, lexer::KeyWordKind> segment;

public:
  SimplePathSegment(Location loc) : Node(loc){};

  void setKeyWord(lexer::KeyWordKind kw) { segment = kw; }
  void setIdentifier(const Identifier id) { segment = id; }

  bool isKeyWord() const {
    return std::holds_alternative<lexer::KeyWordKind>(segment);
  }

  lexer::KeyWordKind getKeyWord() const {
    return std::get<lexer::KeyWordKind>(segment);
  }

  Identifier getName() const { return std::get<Identifier>(segment); }

  std::string asString() const {
    if (isKeyWord()) {
      lexer::KeyWordKind kind = getKeyWord();
      if (kind == lexer::KeyWordKind::KW_SUPER)
        return "super";
      if (kind == lexer::KeyWordKind::KW_SELFVALUE)
        return "self";
      if (kind == lexer::KeyWordKind::KW_CRATE)
        return "crate";
      //if (kind == lexer::KeyWordKind::KW_DOLLARCRATE)
      //  return "$crate";
      assert(false && "unknown keyword");
    }
    return getName().toString();
  }
};

} // namespace rust_compiler::ast
