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

  printf("tryParsePathList\n");

  printTokenState(view);

  if (view.front().getKind() != TokenKind::BraceOpen) {
    return std::nullopt; // failed
  }

  view = view.subspan(1);

  while (view.size() > 4) {
    std::optional<std::shared_ptr<UseTree>> useTree = tryParseUseTree(view);
    if (useTree) {
      printf("tryParseUseTree success\n");
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

    if (view.front().getKind() == TokenKind::Comma) {
      view = view.subspan(1);
      continue;
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

  printf("tryParseUseTree\n");
  printTokenState(view);

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

  if (view.front().getKind() == TokenKind::DoubleColon &&
      view[1].getKind() == TokenKind::BraceOpen) {
    // :: {;
  }

  std::optional<SimplePath> simplePath = tryParseSimplePath(view);
  if (simplePath) {
    printf("tryParseUseTree: found simple path\n");

    view = view.subspan((*simplePath).getTokens());

    if (view[0].getKind() == TokenKind::DoubleColon &&
        view[1].getKind() == TokenKind::Star &&
        view[2].getKind() == TokenKind::SemiColon) {
      // SimplePath :: * ;
    }

    if (view[0].getKind() == TokenKind::DoubleColon &&
        view[1].getKind() == TokenKind::BraceOpen) {
      // SimplePath :: {;
      printf("tryParseUseTree: SimplePath :: {\n");
      view = view.subspan(1);
      std::optional<PathList> pathList = tryParsePathList(view);
      if (pathList) {
        SimplePathDoubleColonWithPathList simple;
        simple.setPathList(*pathList);
        return std::static_pointer_cast<UseTree>(
            std::make_shared<SimplePathDoubleColonWithPathList>(simple));
      }
    }

    // Third line
    if (view[0].getKind() == TokenKind::SemiColon) {
      // SimplePath ;
      SimplePathNode node;
      node.setSimplePath(*simplePath);
      return std::static_pointer_cast<UseTree>(
          std::make_shared<SimplePathNode>(node));
      // UseTree done
    }

    // Rebinding
    if (view.front().isIdentifier() && view[1].isAs() &&
        view[2].isIdentifier() && view[3].getKind() == TokenKind::SemiColon) {
      // SimplePath as Id or _ ;
    }

    if (view.front().getKind() == TokenKind::Comma &&
        view[1].getKind() == TokenKind::BraceClose) {
      SimplePathNode node;
      node.setSimplePath(*simplePath);
      return std::static_pointer_cast<UseTree>(
          std::make_shared<SimplePathNode>(node));
      // UseTree done
    }

    if (view.front().getKind() == TokenKind::BraceClose) {
      SimplePathNode node;
      node.setSimplePath(*simplePath);
      return std::static_pointer_cast<UseTree>(
          std::make_shared<SimplePathNode>(node));
      // UseTree done
    }

    if (view.front().getKind() == TokenKind::Comma) {
      SimplePathNode node;
      node.setSimplePath(*simplePath);
      return std::static_pointer_cast<UseTree>(
          std::make_shared<SimplePathNode>(node));
    }

    return std::nullopt;
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

// FIXME Identifier vs. SimplePath !!!!
