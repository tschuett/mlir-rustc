#pragma once

#include "ADT/Result.h"
#include "AST/LiteralExpression.h"
#include "AST/MetaItemInner.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/TokenStream.h"

#include <memory>
#include <string>
#include <vector>

/// https://doc.rust-lang.org/reference/attributes.html

namespace rust_compiler::ast {

using namespace rust_compiler::adt;

class Literal {
  std::string storage;
  LiteralExpressionKind kind;
};

class MetaItem : public MetaItemInner {};

/// IDENTIFER = (STRING_LITERAL | RAW_STRING_LITERAL)
class MetaNameValueStr : public MetaItem {
  std::string identifier;
  Location loc;

  std::string str;
};

class MetaListPaths : public MetaItem {};

class MetaListNameValueStr : public MetaItem {};

/// IDENTIFIER
class MetaWord : public MetaItem {
  std::string identifier;
  Location loc;

public:
};

class MetaItemSequence : public MetaItem {
  SimplePath path;
  std::vector<std::unique_ptr<MetaItemInner>> sequence;

public:
  MetaItemSequence(SimplePath path,
                   std::vector<std::unique_ptr<MetaItemInner>> sequence)
      : path(std::move(path)), sequence(std::move(sequence)) {}
};

class MetaItemLiteralExpression : public MetaItem {};

class MetaItemPathLit : public MetaItem {};

class MetaItemPath : public MetaItem {
  SimplePath path;

public:
  MetaItemPath(SimplePath path) : path(std::move(path)) {}
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

  Literal parseLiteral();
  SimplePath parseSimplePath();
  SimplePathSegment parseSimplePathSegment();

  lexer::Token peekToken(int i = 0);
  void skipToken(int i = 0);
};

} // namespace rust_compiler::ast
