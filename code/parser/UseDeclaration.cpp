#include "UseDeclaration.h"

#include "AST/UseDeclaration.h"
#include "AST/UseTree.h"
#include "Lexer/Token.h"
#include "SimplePath.h"

#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::use_tree;

namespace rust_compiler::parser {

static std::optional<std::shared_ptr<UseTree>>
tryParseUseTree(std::span<Token> tokens);

std::optional<PathList> tryParsePathList(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  PathList list;

  if (view.front().getKind() != TokenKind::BraceOpen) {
    return std::nullopt; // failed
  }

  view = view.subspan(1);

  while (view.size() > 4) {
    std::optional<std::shared_ptr<UseTree>> useTree = tryParseUseTree(view);
    if (useTree) {
      list.addTree(*useTree);
      view = view.subspan((*useTree)->getTokens());
    }

    // Comma }
    if (view.front().getKind() == TokenKind::Comma &&
        view[1].getKind() == TokenKind::BraceClose) {
      return list;
    }

    // }
    if (view.front().getKind() == TokenKind::BraceClose) {
      return list;
    }

    // failed
    if (view.front().getKind() != TokenKind::Comma) {
      return std::nullopt; // failed!
    }
  }

  return std::nullopt;
}

static std::optional<std::shared_ptr<UseTree>>
tryParseUseTree(std::span<Token> tokens) {
  std::span<Token> view = tokens;

  // First line
  if (view.front().getKind() == TokenKind::Star &&
      view[1].getKind() == TokenKind::SemiColon) {
    // * ;
    std::shared_ptr<UseTree> star =
        std::static_pointer_cast<UseTree>(std::make_shared<Star>(Star()));
    return star;
  }
  if (view.front().getKind() == TokenKind::DoubleColon &&
      view[1].getKind() == TokenKind::Star &&
      view[2].getKind() == TokenKind::SemiColon) {
    // :: * ;
    std::shared_ptr<UseTree> star = std::static_pointer_cast<UseTree>(
        std::make_shared<DoubleColonStar>(DoubleColonStar()));
    return star;
  }

  if (view.front().isIdentifier() &&
      view[1].getKind() == TokenKind::DoubleColon &&
      view[2].getKind() == TokenKind::Star &&
      view[3].getKind() == TokenKind::SemiColon) {
    // SimplePath :: * ;
  }

  // Second line
  if (view.front().isIdentifier() &&
      view[1].getKind() == TokenKind::DoubleColon &&
      view[2].getKind() == TokenKind::BraceOpen) {
    // SimplePath :: {;
  }

  if (view.front().getKind() == TokenKind::DoubleColon &&
      view[1].getKind() == TokenKind::BraceOpen) {
    // :: {;
  }

  if (view.front().getKind() == TokenKind::BraceOpen) {
    // {
    std::optional<PathList> list = tryParsePathList(view);
    if (list) {
      std::shared_ptr<UseTree> star =
          std::static_pointer_cast<UseTree>(std::make_shared<PathList>(*list));
      return star;
    }
    // FIXME
  }

  // Third line
  if (view.front().isIdentifier() &&
      view[1].getKind() == TokenKind::SemiColon) {
    // SimplePath ;
  }

  // Rebinding
  if (view.front().isIdentifier() && view[1].isAs() && view[2].isIdentifier() &&
      view[3].getKind() == TokenKind::SemiColon) {
    // SimplePath as Id or _ ;
  }

  return std::nullopt;
}

std::optional<UseDeclaration> tryParseUseDeclaration(std::span<Token> tokens) {
  std::span<Token> view = tokens;
  UseDeclaration useDeclaration = {view.front().getLocation()};

  if (tokens.front().isUseToken()) {
    view = view.subspan(1);

    std::optional<std::shared_ptr<UseTree>> useTree = tryParseUseTree(view);
    if (useTree) {
      useDeclaration.setComponent(*useTree);
      return useDeclaration;
    }
  }

  return std::nullopt; // FIXME
}

} // namespace rust_compiler::parser
