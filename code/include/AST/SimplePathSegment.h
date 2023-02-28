#pragma once

#include "AST/AST.h"
#include "Lexer/KeyWords.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class SimplePathSegment : public Node {
  std::optional<std::string> identifier;
  std::optional<lexer::KeyWordKind> kind;

public:
  SimplePathSegment(Location loc) : Node(loc){};

  void setKeyWord(lexer::KeyWordKind kw) { kind = kw; }
  void setIdentifier(std::string_view s) { identifier = s; }
};

} // namespace rust_compiler::ast
