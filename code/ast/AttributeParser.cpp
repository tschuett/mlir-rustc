#include "AST/AttributeParser.h"

#include "AST/SimplePath.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;

namespace rust_compiler::ast {

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
      case KeyWordKind::KW_DOLLARCRATE:
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
      default: {
      }
      }
    }
  }

  if (peekToken(1).getKind() == TokenKind::PathSep) {
  }

  if (peekToken(1).getKind() != TokenKind::ParenOpen) {
  }
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

  } else if (peekToken().getKind() == TokenKind::Comma) {
    return std::unique_ptr<MetaItemPath>(new MetaItemPath(std::move(path)));
  } else {
    // report error
  }
}

Literal AttributeParser::parseLiteral() {}

SimplePath AttributeParser::parseSimplePath() {}

SimplePathSegment AttributeParser::parseSimplePathSegment() {}

std::unique_ptr<MetaItemLiteralExpression>
AttributeParser::parseMetaItemLiteralExpression() {}

lexer::Token AttributeParser::peekToken(int i) { return ts.getAt(offset + i); }

void AttributeParser::skipToken(int i) { offset += 1 + i; }

} // namespace rust_compiler::ast
