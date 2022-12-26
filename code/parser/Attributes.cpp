#include "Attributes.h"

#include "AST/InnerAttribute.h"
#include "Token.h"

#include <optional>
#include <sstream>
#include <vector>

namespace rust_compiler {

std::optional<std::string> tryParseLint(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view[0].getKind() == TokenKind::Identifier) {
    if (view[1].getKind() == TokenKind::DoubleColon) {
      if (view[2].getKind() == TokenKind::Identifier) {
        std::stringstream s;
        s << view[0].getIdentifier() << "::" << view[2].getIdentifier();
        return s.str();
      }
    }
  }

  if (view[0].getKind() == TokenKind::Identifier) {
    if (view[1].getKind() != TokenKind::DoubleColon) {
      return view[0].getIdentifier();
    }
  }

  return std::nullopt;
}

std::optional<OuterAttribute> tryParseOuterAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (not(view[0].getKind() == TokenKind::Hash and
          view[1].getKind() == TokenKind::SquareOpen)) {
    return std::nullopt;
  } else {
  }

  return std::nullopt;
}

static std::optional<InnerAttribute>
tryParseWarnAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  if (view[0].getKind() == TokenKind::Hash and
      view[1].getKind() == TokenKind::Exclaim and
      view[2].getKind() == TokenKind::SquareOpen and
      view[3].getKind() == TokenKind::Identifier and
      view[3].getIdentifier() == "warn" and
      view[4].getKind() == TokenKind::ParenOpen) {
    view = view.subspan(5);
  }
  return std::nullopt;
}

static std::optional<InnerAttribute>
tryParseDenyAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  if (view[0].getKind() == TokenKind::Hash and
      view[1].getKind() == TokenKind::Exclaim and
      view[2].getKind() == TokenKind::SquareOpen and
      view[3].getKind() == TokenKind::Identifier and
      view[3].getIdentifier() == "deny" and
      view[4].getKind() == TokenKind::ParenOpen) {
    view = view.subspan(5);
    InnerAttribute attr;
    std::optional<std::string> lint = tryParseLint(view);
    if (lint) {
    }
  }
  return std::nullopt;
}

std::optional<InnerAttribute> tryParseInnerAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (not(view[0].getKind() == TokenKind::Hash and
          view[1].getKind() == TokenKind::Exclaim and
          view[2].getKind() == TokenKind::SquareOpen)) {
    return std::nullopt;
  } else if (view[3].getKind() == TokenKind::Identifier and
             view[3].getIdentifier() == "warn") {
    return tryParseWarnAttribute(tokens);
  } else if (view[3].getKind() == TokenKind::Identifier and
             view[3].getIdentifier() == "deny") {
    return tryParseDenyAttribute(tokens);
  }

  return std::nullopt;
}

std::optional<ClippyAttribute>
tryParseClippyAttribute(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (tokens.front().getKind() == TokenKind::Hash) {
    if (tokens[1].getKind() == TokenKind::Exclaim) {
      if (tokens[2].getKind() == TokenKind::SquareOpen) {
        if (tokens[3].getKind() == TokenKind::Identifier) {
          if (tokens[3].getIdentifier() == "warn" or
              tokens[3].getIdentifier() == "allow" or
              tokens[3].getIdentifier() == "deny") {
            view = view.subspan(4);
            std::vector<std::string> lints;
            do {
              std::optional<std::string> lint = tryParseLint(view);
              if (lint) {
                lints.push_back(*lint);
              }
            } while (view.front().getKind() != TokenKind::Comma);
          }
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace rust_compiler
