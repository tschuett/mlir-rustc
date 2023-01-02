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

std::optional<std::pair<std::string, unsigned>>
tryParseLint(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  if (view[0].getKind() == TokenKind::Identifier) {
    if (view[1].getKind() == TokenKind::DoubleColon) {
      if (view[2].getKind() == TokenKind::Identifier) {
        std::stringstream s;
        s << view[0].getIdentifier() << "::" << view[2].getIdentifier();
        return std::make_pair<std::string, unsigned>(s.str(), 3);
      }
    }
  }

  if (view[0].getKind() == TokenKind::Identifier) {
    if (view[1].getKind() != TokenKind::DoubleColon) {
      return std::make_pair<std::string, unsigned>(view[0].getIdentifier(), 1);
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
    std::optional<std::pair<std::string, unsigned>> lint = tryParseLint(view);
    if (lint) {
      // FIXME
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
              unsigned lintTokens = 0;
              while (view.size() > 1) {
                std::optional<std::pair<std::string, unsigned>> lint =
                    tryParseLint(view);
                if (lint) {
                  lints.push_back(std::get<0>(*lint));
                  view = view.subspan(std::get<1>(*lint));
                  lintTokens += std::get<1>(*lint);
                  if (view.front().getKind() == TokenKind::Comma) {
                    view = view.subspan(1);
                  }
                } else {
                  printf("found clippy\n");
                  return ClippyAttribute(view[0].getLocation(), lints, lintTokens);
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
