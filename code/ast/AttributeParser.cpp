#include "AST/AttributeParser.h"

#include "AST/LiteralExpression.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/Identifier.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "llvm/ADT/StringExtras.h"

#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <vector>

using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;

namespace rust_compiler::ast {

MetaItemInner *MetaItemPath::clone() { return new MetaItemPath(path); }

MetaItemInner *MetaItemSequence::clone() {
  SimplePath newPath = path;
  std::vector<std::shared_ptr<MetaItemInner>> newSequence;

  for (unsigned i = 0; i < sequence.size(); ++i)
    newSequence[i] = std::shared_ptr<MetaItemInner>(sequence[i]->clone());

  return new MetaItemSequence(newPath, newSequence);
}

std::vector<std::shared_ptr<MetaItemInner>>
AttributeParser::parseMetaItemSequence() {
  size_t length = ts.size();
  std::vector<std::shared_ptr<MetaItemInner>> metaItems;

  llvm::errs() << "parseMetaItemSequence: " << length << "\n";
//  for (const Token &token : ts)
//    llvm::errs() << Token2String(token.getKind()) << "\n";

  if (offset < length && peekToken().getKind() != TokenKind::ParenOpen) {
    llvm::errs() << peekToken().getLocation().toString()
                 << "missing ( in delim token tree"
                 << "\n";
    return {};
  }
  skipToken();

  while (offset < length &&
         not(peekToken().getKind() == TokenKind::ParenClose &&
             peekToken(1).getKind() == TokenKind::SquareClose)) {
    if (check(TokenKind::ParenOpen) or check(TokenKind::SquareOpen) or
        check(TokenKind::BraceOpen))
      skipToken();
    std::shared_ptr<MetaItemInner> inner = parseMetaItemInner();
    if (not (bool)inner) {
      return {};
    }
    metaItems.push_back(std::move(inner));

    if (check(TokenKind::ParenClose) or check(TokenKind::SquareClose) or
        check(TokenKind::BraceClose))
      skipToken();

    if (offset < length && peekToken().getKind() != TokenKind::Comma)
      break;

    skipToken();
  }

  if (offset < length && peekToken().getKind() != TokenKind::ParenClose) {
    llvm::errs() << peekToken().getLocation().toString()
                 << ": missing ) in delim token tree"
                 << "\n";
    llvm::errs() << Token2String(peekToken().getKind()) << "\n";
    exit(EXIT_FAILURE);
    return {};
  }

  skipToken();

  llvm::errs() << "parseMetaItemSequence2: " << metaItems.size() << "\n";

  return metaItems;
}

/// MetaItemInner : MetaItem | Expression (literal)
std::shared_ptr<MetaItemInner> AttributeParser::parseMetaItemInner() {
  llvm::errs() << "parseMetaItemInner"
               << "\n";

  if (not peekToken().isIdentifier()) {
    if (peekToken().isKeyWord()) {
      switch (peekToken().getKeyWordKind()) {
      case KeyWordKind::KW_SUPER:
      case KeyWordKind::KW_SELFVALUE:
      case KeyWordKind::KW_CRATE:
        return parsePathMetaItem();
      case KeyWordKind::KW_TRUE:
      case KeyWordKind::KW_FALSE:
        return parseMetaItemLiteralExpression();
      default: {
      }
      }
    } else {
      switch (peekToken().getKind()) {
      case TokenKind::CHAR_LITERAL:
      case TokenKind::STRING_LITERAL:
      case TokenKind::BYTE_STRING_LITERAL:
      case TokenKind::INTEGER_LITERAL:
      case TokenKind::FLOAT_LITERAL:
        return parseMetaItemLiteralExpression();
      case TokenKind::PathSep:
        return parsePathMetaItem();
      case TokenKind::Dollar:
        if (peekToken(1).isKeyWord() &&
            (peekToken(1).getKeyWordKind() == KeyWordKind::KW_CRATE))
          return parsePathMetaItem();
        break;
      default: {
        llvm::errs() << "unknown token :" << Token2String(peekToken().getKind())
                     << " in parseMetaItemInner"
                     << "\n";
      }
      }
    }
  }

  if (peekToken(1).getKind() == TokenKind::PathSep)
    return parsePathMetaItem();

  Identifier ident = peekToken().getIdentifier();
  Location identLoc = peekToken().getLocation();

  if (isMetaItemEnd(peekToken(1).getKind())) {
    skipToken();

    return std::shared_ptr<MetaWord>(new MetaWord(ident, identLoc));
  }

  if (peekToken(1).getKind() == TokenKind::Eq) {
    if (peekToken(2).getKind() == TokenKind::STRING_LITERAL &&
        isMetaItemEnd(peekToken(3).getKind())) {
      Token valueToken = peekToken(2);
      Location loc = valueToken.getLocation();

      skipToken(2);

      return std::shared_ptr<MetaNameValueString>(
          new MetaNameValueString(ident, identLoc, valueToken));
    } else {
      return parsePathMetaItem();
    }
  }

  if (peekToken(1).getKind() != TokenKind::ParenOpen) {
    // report error
    llvm::errs() << llvm::formatv(
                        "unexpected token {0} after identifier in attribute",
                        Token2String(peekToken(1).getKind()))
                 << "\n";
    return nullptr;
  }

  if (peekToken().getKind() == TokenKind::Identifier)
    return parsePathMetaItem();

  std::vector<std::shared_ptr<MetaItemInner>> metaItems =
      parseMetaItemSequence();

  // try meta name value string
  std::vector<MetaNameValueString> metaNameValueStringItems;
  for (const auto &item : metaItems) {
    std::optional<std::shared_ptr<MetaNameValueString>> convertedItem =
        item->tryMetaNameValueString();
    if (!convertedItem) {
      metaNameValueStringItems.clear();
      break;
    }
  }

  if (!metaItems.empty())
    return std::shared_ptr<MetaListNameValueString>(
        new MetaListNameValueString(ident, identLoc, metaNameValueStringItems));

  // try meta list paths
  std::vector<SimplePath> pathItems;
  for (const auto &item : metaItems) {
    std::optional<SimplePath> convertedPath = item->tryPathItem();
    if (!convertedPath) {
      pathItems.clear();
      break;
    }
    pathItems.push_back(*convertedPath);
  }

  if (!pathItems.empty())
    return std::shared_ptr<MetaListPaths>(
        new MetaListPaths(ident, identLoc, pathItems));

  llvm::errs() << "failed to parse any meta item inner"
               << "\n";

  return nullptr;
}

std::shared_ptr<MetaItem> AttributeParser::parsePathMetaItem() {
  std::optional<SimplePath> path = parseSimplePath();
  if (!path) {
    // report error
    llvm::errs() << peekToken().getLocation().toString()
                 << "failed to parse simple path in attribute"
                 << "\n";
    return nullptr;
  }

  if (path->isEmpty()) {
    // report error
    llvm::errs() << peekToken().getLocation().toString()
                 << "failed to parse simple path in attribute"
                 << "\n";
    return nullptr;
  }

  if (peekToken().getKind() == TokenKind::ParenOpen) {
    std::vector<std::shared_ptr<MetaItemInner>> metaItems =
        parseMetaItemSequence();

    return std::shared_ptr<MetaItemSequence>(
        new MetaItemSequence(*path, std::move(metaItems)));
  } else if (peekToken().getKind() == TokenKind::Eq) {
    skipToken();

    Location loc = peekToken().getLocation();
    std::optional<AttributeLiteral> lit = parseLiteral();

    if (!lit) {
      // report error
      llvm::errs() << "failed to parse literal in attribute"
                   << "\n";
      return nullptr;
    }
    AttributeLiteralExpression expr = {std::move(*lit), loc};

    return std::shared_ptr<MetaItemPathLit>(
        new MetaItemPathLit(*path, std::move(expr)));
  } else if (peekToken().getKind() == TokenKind::Comma) {
    return std::shared_ptr<MetaItemPath>(new MetaItemPath(*path));
  }

  // report error
  llvm::errs() << llvm::formatv("unrecognized token {0} in meta item",
                                Token2String(peekToken().getKind()))
               << "\n";

  return nullptr;
}

std::optional<AttributeLiteral> AttributeParser::parseLiteral() {
  if (peekToken().isKeyWord()) {
    switch (peekToken().getKeyWordKind()) {
    case KeyWordKind::KW_TRUE:
      return AttributeLiteral("true", LiteralExpressionKind::True);
    case KeyWordKind::KW_FALSE:
      return AttributeLiteral("false", LiteralExpressionKind::False);
    default: {
    }
    }
  } else {
    switch (peekToken().getKind()) {
    case TokenKind::CHAR_LITERAL:
      return AttributeLiteral(peekToken().getUtf8Storage(),
                              LiteralExpressionKind::CharLiteral);
    case TokenKind::STRING_LITERAL:
      return AttributeLiteral(peekToken().getUtf8Storage(),
                              LiteralExpressionKind::StringLiteral);
    case TokenKind::BYTE_LITERAL:
      return AttributeLiteral(peekToken().getLiteral(),
                              LiteralExpressionKind::ByteLiteral);
    case TokenKind::BYTE_STRING_LITERAL:
      return AttributeLiteral(peekToken().getLiteral(),
                              LiteralExpressionKind::ByteStringLiteral);
    case TokenKind::INTEGER_LITERAL:
      return AttributeLiteral(peekToken().getLiteral(),
                              LiteralExpressionKind::IntegerLiteral);
    case TokenKind::FLOAT_LITERAL:
      return AttributeLiteral(peekToken().getLiteral(),
                              LiteralExpressionKind::FloatLiteral);
    default: {
    }
    }
  }
  llvm::errs() << llvm::formatv("expected literal found {0}",
                                Token2String(peekToken().getKind()))
               << "\n";
  return std::nullopt;
}

std::optional<SimplePath> AttributeParser::parseSimplePath() {
  Location loc = peekToken().getLocation();
  SimplePath path{loc};

  if (peekToken().getKind() == TokenKind::PathSep) {
    path.setWithDoubleColon();
    skipToken();
  }

  std::optional<SimplePathSegment> segment = parseSimplePathSegment();
  if (!segment) {
    // report error
    llvm::errs() << "failed to parse simple tpath in attribute simple path"
                 << "\n";
    return std::nullopt;
  }
  path.addPathSegment(*segment);

  while (peekToken().getKind() == TokenKind::PathSep) {
    skipToken();

    std::optional<SimplePathSegment> segment = parseSimplePathSegment();
    if (!segment) {
      // report error
      llvm::errs()
          << "failed to parse simple path segment in attribute simple path"
          << "\n";
      return std::nullopt;
    }
    path.addPathSegment(*segment);
  }

  return SimplePath(loc);
}

std::optional<SimplePathSegment> AttributeParser::parseSimplePathSegment() {
  Token token = peekToken();

  SimplePathSegment segment = {token.getLocation()};
  if (peekToken().isKeyWord()) {
    switch (peekToken().getKeyWordKind()) {
    case KeyWordKind::KW_SUPER: {
      skipToken();
      segment.setKeyWord(KeyWordKind::KW_SUPER);
      return segment;
    }
    case KeyWordKind::KW_SELFVALUE: {
      skipToken();
      segment.setKeyWord(KeyWordKind::KW_SELFVALUE);
      return segment;
    }
    case KeyWordKind::KW_CRATE: {
      skipToken();
      segment.setKeyWord(KeyWordKind::KW_CRATE);
      return segment;
    }
    default: {
      llvm::errs() << "unknown token: " << Token2String(peekToken().getKind())
                   << "in parseSimplePathSegment"
                   << "\n";
    }
    }
  } else {
    switch (peekToken().getKind()) {
    case TokenKind::Identifier: {
      skipToken();
      segment.setIdentifier(token.getIdentifier());
      return segment;
    }
    case TokenKind::Dollar:
      if (peekToken(1).isKeyWord() &&
          (peekToken(1).getKeyWordKind() == KeyWordKind::KW_CRATE)) {
        skipToken(1);
        segment.setIdentifier(Identifier("$crate"));
        return segment;
      }
      break;
    default: {
      llvm::errs() << "unknown token: " << Token2String(peekToken().getKind())
                   << "in parseSimplePathSegment"
                   << "\n";
    }
    }
  }

  llvm::errs() << llvm::formatv("unexpected token {0} in simple path segment",
                                Token2String(token.getKind()))
               << "\n";

  return std::nullopt;
}

std::shared_ptr<MetaItemLiteralExpression>
AttributeParser::parseMetaItemLiteralExpression() {
  Location loc = peekToken().getLocation();
  auto lit = parseLiteral();
  if (!lit)
    return nullptr;

  // eat the literal
  skipToken();

  AttributeLiteralExpression expr = {*lit, loc};
  return std::shared_ptr<MetaItemLiteralExpression>(
      new MetaItemLiteralExpression(expr));
}

lexer::Token AttributeParser::peekToken(int i) { return ts[offset + i]; }

void AttributeParser::skipToken(int i) { offset += 1 + i; }

bool AttributeParser::isMetaItemEnd(TokenKind kind) {
  return kind == TokenKind::Comma || kind == TokenKind::ParenClose;
}

MetaItemInner *MetaWord::clone() { return new MetaWord(identifier, loc); }

MetaItemInner *MetaNameValueString::clone() {
  return new MetaNameValueString(identifier, loc, str);
}

MetaItemInner *MetaListPaths::clone() {
  return new MetaListPaths(ident, loc, paths);
}

MetaItemInner *MetaListNameValueString::clone() {
  return new MetaListNameValueString(ident, loc, kvs);
}

MetaItemInner *MetaItemPathLit::clone() {
  return new MetaItemPathLit(path, expr);
}

MetaItemInner *MetaItemLiteralExpression::clone() {
  return new MetaItemLiteralExpression(expr);
}

std::optional<std::shared_ptr<MetaNameValueString>>
MetaNameValueString::tryMetaNameValueString() const {
  return std::shared_ptr<MetaNameValueString>(
      new MetaNameValueString(identifier, loc, str));
}

} // namespace rust_compiler::ast
