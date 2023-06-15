#pragma once

#include "ADT/Result.h"
#include "ADT/Utf8String.h"
#include "AST/LiteralExpression.h"
#include "AST/MetaItemInner.h"
#include "AST/SimplePath.h"
#include "AST/SimplePathSegment.h"
#include "Lexer/Identifier.h"
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
  Utf8String utf8Storage;
  LiteralExpressionKind kind;

public:
  AttributeLiteral(std::string_view lit, LiteralExpressionKind kind)
      : storage(lit), kind(kind){};

  AttributeLiteral(const Utf8String &lit, LiteralExpressionKind kind)
      : utf8Storage(lit), kind(kind){};

  LiteralExpressionKind getKind() const { return kind; }
};

class AttributeLiteralExpression {
  AttributeLiteral literal;
  Location loc;

public:
  AttributeLiteralExpression(AttributeLiteral lit, Location loc)
      : literal(lit), loc(loc){};
};

enum class MetaItemKind {
  MetaNameValueString,
  MetaListPaths,
  MetaListNameValueString,
  MetaWord,
  MetaItemSequence,
  MetaLiteralExpression,
  MetaItemPathLit,
  MetaItemPath
};

class MetaItem : public MetaItemInner {
  MetaItemKind kind;

public:
  MetaItem(MetaItemKind kind) : kind(kind) {}
  bool isMetaItem() const override { return true; }

  MetaItemKind getKind() const { return kind; }
};

/// IDENTIFER = (STRING_LITERAL | RAW_STRING_LITERAL)
class MetaNameValueString : public MetaItem {
  Identifier identifier;
  Location loc;

  Token str;

public:
  MetaNameValueString(const Identifier &key, Location lc, const Token &t)
      : MetaItem(MetaItemKind::MetaNameValueString), identifier(key), loc(lc),
        str(t) {}

  MetaItemInner *clone() override;

  std::optional<std::shared_ptr<MetaNameValueString>>
  tryMetaNameValueString() const override;

  bool isKeyValuePair() const override { return true; }
};

class MetaListPaths : public MetaItem {
  Identifier ident;
  Location loc;
  std::vector<SimplePath> paths;

public:
  MetaListPaths(const Identifier &id, Location lc, std::span<SimplePath> path)
      : MetaItem(MetaItemKind::MetaListPaths), ident(id), loc(lc) {
    paths = {path.begin(), path.end()};
  }

  MetaItemInner *clone() override;
  bool isKeyValuePair() const override { return false; }
};

class MetaListNameValueString : public MetaItem {
  Identifier ident;
  Location loc;
  std::vector<MetaNameValueString> kvs;

public:
  MetaListNameValueString(const Identifier &id, Location lc,
                          std::span<MetaNameValueString> kv)
      : MetaItem(MetaItemKind::MetaListNameValueString), ident(id), loc(lc) {
    kvs = {kv.begin(), kv.end()};
  }

  MetaItemInner *clone() override;
  bool isKeyValuePair() const override { return false; }
};

/// IDENTIFIER
class MetaWord : public MetaItem {
  Identifier identifier;
  Location loc;

public:
  MetaWord(const Identifier &word, Location loc)
      : MetaItem(MetaItemKind::MetaWord), identifier(word), loc(loc) {}

  MetaItemInner *clone() override;

  bool isKeyValuePair() const override { return false; }
  Identifier getIdentifier() const { return identifier; }
};

class MetaItemSequence : public MetaItem {
  SimplePath path;
  std::vector<std::shared_ptr<MetaItemInner>> sequence;

public:
  MetaItemSequence(SimplePath path,
                   std::vector<std::shared_ptr<MetaItemInner>> sequence)
      : MetaItem(MetaItemKind::MetaItemSequence), path(std::move(path)),
        sequence(std::move(sequence)) {}

  MetaItemInner *clone() override;

  bool isKeyValuePair() const override { return false; }
};

class MetaItemLiteralExpression : public MetaItem {
  AttributeLiteralExpression expr;

public:
  MetaItemLiteralExpression(AttributeLiteralExpression exp)
      : MetaItem(MetaItemKind::MetaLiteralExpression), expr(exp) {}
  MetaItemInner *clone() override;

  bool isKeyValuePair() const override { return false; }
};

class MetaItemPathLit : public MetaItem {
  SimplePath path;
  AttributeLiteralExpression expr;

public:
  MetaItemPathLit(SimplePath p, AttributeLiteralExpression exp)
      : MetaItem(MetaItemKind::MetaItemPathLit), path(p), expr(exp){};

  MetaItemInner *clone() override;

  bool isKeyValuePair() const override { return false; }
};

class MetaItemPath : public MetaItem {
  SimplePath path;

public:
  MetaItemPath(SimplePath path)
      : MetaItem(MetaItemKind::MetaItemPath), path(path) {}

  MetaItemInner *clone() override;

  bool isKeyValuePair() const override { return false; }
};

class AttributeParser {
  std::vector<Token> ts;

  size_t offset;

public:
  AttributeParser(const std::vector<Token> &ts) : ts(ts), offset(0) {}

  std::vector<std::shared_ptr<MetaItemInner>> parseMetaItemSequence();

private:
  std::shared_ptr<MetaItemInner> parseMetaItemInner();
  std::shared_ptr<MetaItem> parsePathMetaItem();

  std::shared_ptr<MetaItemLiteralExpression> parseMetaItemLiteralExpression();

  std::optional<AttributeLiteral> parseLiteral();
  std::optional<SimplePath> parseSimplePath();
  std::optional<SimplePathSegment> parseSimplePathSegment();

  Token peekToken(int i = 0);
  void skipToken(int i = 0);

  bool isMetaItemEnd(TokenKind kind);

  bool check(TokenKind kind) { return peekToken().getKind() == kind; }
};

} // namespace rust_compiler::ast
