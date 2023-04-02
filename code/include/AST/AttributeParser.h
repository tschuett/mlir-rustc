#pragma once

#include "ADT/Result.h"
#include "AST/LiteralExpression.h"
#include "AST/MetaItemInner.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

/// https://doc.rust-lang.org/reference/attributes.html

namespace rust_compiler::ast {

using namespace rust_compiler::adt;
using namespace rust_compiler::lexer;

class AttributeLiteral {
  std::string storage;

public:
  AttributeLiteral(std::string_view, LiteralExpressionKind);
};

class AttributeLiteralExpression {
public:
  AttributeLiteralExpression(AttributeLiteral, Location);
};

class MetaItem : public MetaItemInner {};

/// IDENTIFER = (STRING_LITERAL | RAW_STRING_LITERAL)
class MetaNameValueString : public MetaItem {
  std::string identifier;
  Location loc;

  std::string str;

public:
  MetaNameValueString(std::string_view key, Location, std::string_view,
                      Location);

  MetaItemInner *clone() override;
};

class MetaListPaths : public MetaItem {
public:
  MetaListPaths(std::string_view, Location, std::span<SimplePath>);

  MetaItemInner *clone() override;
};

class MetaListNameValueString : public MetaItem {
public:
  MetaListNameValueString(std::string_view, Location,
                          std::span<MetaNameValueString>);

  MetaItemInner *clone() override;
};

/// IDENTIFIER
class MetaWord : public MetaItem {
  std::string identifier;
  Location loc;

public:
  MetaWord(std::string_view word, Location loc);

  MetaItemInner *clone() override;
};

class MetaItemSequence : public MetaItem {
  SimplePath path;
  std::vector<std::unique_ptr<MetaItemInner>> sequence;

public:
  MetaItemSequence(SimplePath path,
                   std::vector<std::unique_ptr<MetaItemInner>> sequence)
      : path(std::move(path)), sequence(std::move(sequence)) {}

  MetaItemInner *clone() override;
};

class MetaItemLiteralExpression : public MetaItem {};

class MetaItemPathLit : public MetaItem {
public:
  MetaItemPathLit(SimplePath, AttributeLiteralExpression);

  MetaItemInner *clone() override;
};

class MetaItemPath : public MetaItem {
  SimplePath path;

public:
  MetaItemPath(SimplePath path) : path(std::move(path)) {}

  MetaItemInner *clone() override;
};

class AttributeParser {
  lexer::TokenStream ts;

  size_t offset;

public:
  AttributeParser(const lexer::TokenStream &ts) : ts(ts), offset(0) {}

  std::vector<std::unique_ptr<MetaItemInner>> parseMetaItemSequence();

private:
  std::unique_ptr<MetaItemInner> parseMetaItemInner();
  std::unique_ptr<MetaItem> parsePathMetaItem();

  std::unique_ptr<MetaItemLiteralExpression> parseMetaItemLiteralExpression();

  std::optional<AttributeLiteral> parseLiteral();
  SimplePath parseSimplePath();
  std::optional<SimplePathSegment> parseSimplePathSegment();

  Token peekToken(int i = 0);
  void skipToken(int i = 0);

  bool isMetaItemEnd(TokenKind kind);
};

} // namespace rust_compiler::ast
