#include "AST/AttributeParser.h"

#include "AST/LiteralExpression.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;

namespace rust_compiler::ast {

static std::string unquoteString(std::string input) {
  assert(input.front() == '"');
  assert(input.back() == '"');
  return input.substr(1, input.length() - 2);
}

MetaItemInner *MetaItemPath::clone() { return new MetaItemPath(path); }

MetaItemInner *MetaItemSequence::clone() {
  SimplePath newPath = path;
  std::vector<std::unique_ptr<MetaItemInner>> newSequence;

  for (unsigned i = 0; i < sequence.size(); ++i)
    newSequence[i] = std::unique_ptr<MetaItemInner>(sequence[i]->clone());

  return new MetaItemSequence(newPath, std::move(newSequence));
}

std::vector<std::unique_ptr<MetaItemInner>>
AttributeParser::parseMetaItemSequence() {
  size_t length = ts.getLength();
  std::vector<std::unique_ptr<MetaItemInner>> metaItems;

  if (peekToken().getKind() != TokenKind::ParenOpen) {
    llvm::errs() << peekToken().getLocation().toString()
                 << "missing ( in delim token tree"
                 << "\n";
    return {};
  }
  skipToken();

  while (offset < length && peekToken().getKind() != TokenKind::ParenClose) {
    std::unique_ptr<MetaItemInner> inner = parseMetaItemInner();
    if (inner == nullptr) {
      return {};
    }
    metaItems.push_back(std::move(inner));

    if (peekToken().getKind() != TokenKind::Comma)
      break;

    skipToken();
  }

  if (peekToken().getKind() != TokenKind::ParenClose) {
    llvm::errs() << peekToken().getLocation().toString()
                 << ": missing ) in delim token tree"
                 << "\n";
    return {};
  }

  skipToken();

  return metaItems;
}

std::unique_ptr<MetaItemInner> AttributeParser::parseMetaItemInner() {
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
      }
      }
    }
  }

  if (peekToken(1).getKind() == TokenKind::PathSep)
    return parsePathMetaItem();

  std::string ident = peekToken().toString();
  Location identLoc = peekToken().getLocation();

  if (isMetaItemEnd(peekToken(1).getKind())) {
    skipToken();

    return std::unique_ptr<MetaWord>(new MetaWord(ident, identLoc));
  }

  if (peekToken(1).getKind() == TokenKind::Eq) {
    if (peekToken(2).getKind() == TokenKind::STRING_LITERAL &&
        isMetaItemEnd(peekToken(3).getKind())) {
      Token valueToken = peekToken(2);
      std::string value = valueToken.toString();
      Location loc = valueToken.getLocation();

      skipToken(2);

      std::string rawValue = unquoteString(value);

      return std::unique_ptr<MetaNameValueString>(
          new MetaNameValueString(ident, identLoc, rawValue, loc));
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

  std::vector<std::unique_ptr<MetaItemInner>> metaItems =
      parseMetaItemSequence();

  // try meta name value string
  std::vector<MetaNameValueString> metaNameValueStringItems;
  for (const auto &item : metaItems) {
    std::unique_ptr<MetaNameValueString> convertedItem =
        item->tryMetaNameValueString();
    if (convertedItem == nullptr) {
      metaNameValueStringItems.clear();
      break;
    }
  }

  if (!metaItems.empty())
    return std::unique_ptr<MetaListNameValueString>(
        new MetaListNameValueString(ident, identLoc, metaNameValueStringItems));

  // try meta list paths
  std::vector<SimplePath> pathItems;
  for (const auto &item : metaItems) {
    SimplePath convertedPath = item->tryPathItem();
    if (convertedPath.isEmpty()) {
      pathItems.clear();
      break;
    }
    pathItems.push_back(std::move(convertedPath));
  }

  if (!pathItems.empty())
    return std::unique_ptr<MetaListPaths>(
        new MetaListPaths(ident, identLoc, pathItems));

  llvm::errs() << "failed to parse any meta item inner"
               << "\n";

  return nullptr;
}

std::unique_ptr<MetaItem> AttributeParser::parsePathMetaItem() {
  SimplePath path = parseSimplePath();
  if (path.isEmpty()) {
    // report error
    llvm::errs() << peekToken().getLocation().toString()
                 << "failed to parse simple path in attribute"
                 << "\n";
    return nullptr;
  }

  if (peekToken().getKind() == TokenKind::ParenOpen) {
    std::vector<std::unique_ptr<MetaItemInner>> metaItems =
        parseMetaItemSequence();

    return std::unique_ptr<MetaItemSequence>(
        new MetaItemSequence(std::move(path), std::move(metaItems)));
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

    return std::unique_ptr<MetaItemPathLit>(
        new MetaItemPathLit(std::move(path), std::move(expr)));
  } else if (peekToken().getKind() == TokenKind::Comma) {
    return std::unique_ptr<MetaItemPath>(new MetaItemPath(std::move(path)));
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
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::CharLiteral);
    case TokenKind::STRING_LITERAL:
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::StringLiteral);
    case TokenKind::BYTE_LITERAL:
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::ByteLiteral);
    case TokenKind::BYTE_STRING_LITERAL:
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::ByteStringLiteral);
    case TokenKind::INTEGER_LITERAL:
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::IntegerLiteral);
    case TokenKind::FLOAT_LITERAL:
      return AttributeLiteral(peekToken().toString(),
                              LiteralExpressionKind::FloatLiteral);
    default: {
    }
    }
  }
  llvm::errs() << llvm::formatv("expected literal found {0}",
                                Token2String(peekToken().getKind()))
               << "\n";
  return std::nullopt;
  ;
}

SimplePath AttributeParser::parseSimplePath() {}

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
        segment.setIdentifier(Identifier::fromString("$crate"));
        return segment;
      }
      break;
    default: {
    }
    }
  }

  llvm::errs() << llvm::formatv("unexpected token {0} in simple path segment",
                                Token2String(token.getKind()))
               << "\n";

  return std::nullopt;
}

std::unique_ptr<MetaItemLiteralExpression>
AttributeParser::parseMetaItemLiteralExpression() {}

lexer::Token AttributeParser::peekToken(int i) { return ts.getAt(offset + i); }

void AttributeParser::skipToken(int i) { offset += 1 + i; }

bool AttributeParser::isMetaItemEnd(TokenKind kind) {
  return kind == TokenKind::Comma || kind == TokenKind::ParenClose;
}

} // namespace rust_compiler::ast
