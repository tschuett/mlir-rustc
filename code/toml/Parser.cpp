#include "Toml/Parser.h"

#include "Toml/Array.h"
#include "Toml/InlineTable.h"
#include "Toml/KeyValuePair.h"
#include "Toml/Token.h"

#include <memory>
#include <optional>

namespace rust_compiler::toml {

static std::optional<Array> tryParseArray(std::span<Token> tab) {
  std::span<Token> view = tab;

  Array array;

  if (view.size() < 3)
    return std::nullopt;

  if (view.front().getKind() == TokenKind::SquareOpen) {
    view = view.subspan(1);
    while (view.size() > 2) {
      if (view.front().getKind() == TokenKind::String &&
          view[1].getKind() == TokenKind::Comma) {
        array.addElement(view.front().getString());
        view = view.subspan(2);
      } else if (view.front().getKind() == TokenKind::String &&
                 view[1].getKind() == TokenKind::SquareClose) {
        array.addElement(view.front().getString());
        view = view.subspan(2);
        return array;
      } else {
        return std::nullopt;
      }
    }
  }

  return std::nullopt;
}

static std::optional<InlineTable> tryParseInlineTable(std::span<Token> tab);

// static std::optional<std::pair<std::string, std::vector<std::string>>>
// tryParseKeyArrayPair(std::span<Token> view) {
//   std::span<Token> pair = view;
//
//   if (pair.size() < 3)
//     return std::nullopt;
// }

static std::optional<KeyValuePair> tryParseKeyValuePair(std::span<Token> view) {
  std::span<Token> pair = view;
  std::string left;
  std::string right;

  if (pair.size() < 3)
    return std::nullopt;

  if (pair.front().getKind() == TokenKind::Identifier) {
    left = pair.front().getString();
    pair = pair.subspan(1);
    if (pair.front().getKind() == TokenKind::Equal) {
      pair = pair.subspan(1);
      if (pair.front().getKind() == TokenKind::String) {
        right = pair.front().getString();
        pair = pair.subspan(1);
        return KeyValuePair(left, right);
      } else if (pair.front().getKind() == TokenKind::BraceOpen) {
        std::optional<InlineTable> inlineTable = tryParseInlineTable(pair);
        if (inlineTable) {
          pair = pair.subspan(inlineTable->getNrOfTokens());
          // FIXME
          return KeyValuePair(left,
                              static_pointer_cast<Value>(
                                  std::make_shared<InlineTable>(*inlineTable)));
        } else {
          // FIXME
          return std::nullopt;
        }
      } else if (pair.front().getKind() == TokenKind::SquareOpen) {
        std::optional<Array> array = tryParseArray(pair);
        if (array) {
          return KeyValuePair(left, static_pointer_cast<Value>(
                                        std::make_shared<Array>(*array)));
        }
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

static std::optional<InlineTable> tryParseInlineTable(std::span<Token> tab) {
  std::span<Token> view = tab;
  InlineTable table;
  // size_t tokens = 0;

  if (view.size() < 3)
    return std::nullopt;

  if (view.front().getKind() == TokenKind::BraceOpen) {
    view = view.subspan(1);
    while (view.size() > 3) {
      std::optional<KeyValuePair> kv = tryParseKeyValuePair(view);
      if (kv) {
        printf("found inline key pair: %s\n", kv->toString().c_str());
        printf("found inline key pair: %lu\n", kv->getNrOfTokens());
        // tokens += kv->getNrOfTokens();
        view = view.subspan(kv->getNrOfTokens());
        table.addPair(std::make_shared<KeyValuePair>(*kv));
        if (view.front().getKind() == TokenKind::BraceClose) { // ?
          printf("returned table\n");
          return table;
        } else if (view.front().getKind() == TokenKind::Comma) { // ?
          printf("found comma\n");
          view = view.subspan(1); // Comma
          continue;
          // else if (view.front().getKind() != TokenKind::Comma) { // ?
          // printf("found gave up: %s\n", view[0].toString().c_str());
          // return std::nullopt;
        }
      } else if (view.front().getKind() == TokenKind::BraceClose) {
        // tokens += 1; // Brace close
        return table;
      } else {
        printf("inline failed because %s \n", view[0].toString().c_str());
        return std::nullopt;
      }
    }

    printf("inline failed2 because %s \n", view[0].toString().c_str());
    return std::nullopt;
  } else {
    printf("inline failed3 because %s \n", view[0].toString().c_str());
    return std::nullopt;
  }
}

static std::optional<std::string> tryParseHeader(std::span<Token> view) {
  if (view.size() < 4)
    return std::nullopt;

  std::span<Token> header = view;
  if (header.front().getKind() == TokenKind::SquareOpen) {
    header = header.subspan(1);
    if (header.front().getKind() == TokenKind::Identifier) {
      Token id = header.front();
      header = header.subspan(1);
      if (header.front().getKind() == TokenKind::SquareClose) {
        return id.getString();
      }
    }
  }

  return std::nullopt;
}

static std::optional<std::pair<Table, size_t>>
tryParseTable(std::span<Token> view) {
  std::span<Token> tab = view;
  Table table;
  size_t tokens = 0;

  std::optional<std::string> header = tryParseHeader(tab);
  if (not header)
    return std::nullopt;
  table.setHeader(*header);
  tab = tab.subspan(3); // [ string ]
  tokens += 3;

  while (tab.size() > 0) {
    std::optional<KeyValuePair> kv = tryParseKeyValuePair(tab);
    if (kv) {
      tab = tab.subspan(kv->getNrOfTokens()); // 2* string + Equal Token
      tokens += kv->getNrOfTokens();
      table.addPair(std::make_shared<KeyValuePair>(*kv));
    } else {
      return std::make_pair<Table, size_t>(std::move(table), std::move(tokens));
    }
  }

  return std::nullopt;
}

std::optional<Toml> tryParse(TokenStream ts) {
  Toml toml;
  std::span<Token> view = ts.getViewAt(0);

  printf("tryParse %lu\n", view.size());

  while (view.size() > 1) {
    printf("view: %lu\n", view.size());
    std::optional<std::string> header = tryParseHeader(view);
    if (header) {
      std::optional<std::pair<Table, size_t>> table = tryParseTable(view);
      if (table) {
        std::pair<Table, size_t> tab = *table;
        toml.addTable(std::make_shared<Table>(std::get<0>(tab)));
        view = view.subspan(std::get<1>(tab));
      } else {
        // consume
        printf("no table\n");
      }
    } else {
      std::optional<KeyValuePair> pair = tryParseKeyValuePair(view);
      if (pair) {
        view = view.subspan(pair->getNrOfTokens()); // 2*string + Equal Token
        toml.addKeyValuePair(std::make_shared<KeyValuePair>(*pair));
      } else {
        printf("no pair: %s\n", view[0].toString().c_str());
        printf("no pair: %s\n", view[1].toString().c_str());
      }
    }
  }

  printf("tryParse failed\n");

  return std::nullopt;
}

} // namespace rust_compiler::toml

// https://toml.io/en/v1.0.0
