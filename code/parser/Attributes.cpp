#include "Attributes.h"

#include "AST/ClippyAttribute.h"
#include "AST/InnerAttribute.h"
#include "Lexer/Token.h"
#include "mlir/IR/Location.h"

#include <optional>
#include <sstream>
#include <vector>

namespace rust_compiler {

using namespace rust_compiler::lexer;

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
    // InnerAttribute attr;
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
tryParseClippyAttribute(mlir::Location location, std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view.front().getKind() == TokenKind::Hash) {
    if (view[1].getKind() == TokenKind::Exclaim) {
      if (view[2].getKind() == TokenKind::SquareOpen) {
        if (view[3].getKind() == TokenKind::Identifier) {
          if (view[3].getIdentifier() == "warn" or
              view[3].getIdentifier() == "allow" or
              view[3].getIdentifier() == "deny") {
            if (view[4].getKind() == TokenKind::ParenOpen) {
              view = view.subspan(5);
              std::vector<std::string> lints;
              while (view.size() > 1) {
                std::optional<std::string> lint = tryParseLint(view);
                if (lint) {
                  lints.push_back(*lint);
                  view = view.subspan(1);
                } else if (view.front().getKind() == TokenKind::Comma) {
                  printf("found clippy\n");
                  return ClippyAttribute(location, lints);
                }
              }
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace rust_compiler
