#pragma once

#include "AST/AST.h"
#include "Lexer/KeyWords.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace rust_compiler::ast {

class SimplePathSegment : public Node {
  std::variant<std::string, lexer::KeyWordKind> segment;

public:
  SimplePathSegment(Location loc) : Node(loc){};

  void setKeyWord(lexer::KeyWordKind kw) { segment = kw; }
  void setIdentifier(std::string_view s) { segment = std::string(s); }

  bool isKeyWord() const {
    return std::holds_alternative<lexer::KeyWordKind>(segment);
  }

  lexer::KeyWordKind getKeyWord() const {
    return std::get<lexer::KeyWordKind>(segment);
  }

  std::string getName() const { return std::get<std::string>(segment); }

  std::string asString() const {
    if (isKeyWord()) {
      lexer::KeyWordKind kind = getKeyWord();
      if (kind == lexer::KeyWordKind::KW_SUPER)
        return "super";
      if (kind == lexer::KeyWordKind::KW_SELFVALUE)
        return "self";
      if (kind == lexer::KeyWordKind::KW_CRATE)
        return "crate";
      if (kind == lexer::KeyWordKind::KW_DOLLARCRATE)
        return "$crate";
      assert(false && "unknown keyword");
    }
    return getName();
  }
};

} // namespace rust_compiler::ast
