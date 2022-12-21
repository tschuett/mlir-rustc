#include "Toml/Parser.h"

#include "Toml/Token.h"

#include <optional>

namespace rust_compiler::toml {

static std::optional<std::pair<std::string, std::string>>
tryParseKeyValuePair(std::span<Token> view) {
  return std::nullopt;
}

static std::optional<std::string> tryParseHeader(std::span<Token> view) {
  std::span<Token> header = view;
  if (header.front().getKind() == TokenKind::BraceOpen) {
    header = header.subspan(1);
    if (header.front().getKind() == TokenKind::Identifier) {
      Token id = header.front();
      header = header.subspan(1);
      if (header.front().getKind() == TokenKind::BraceClose) {
        return id.getString();
      }
    }
  }

  return std::nullopt;
}

static std::optional<std::pair<Table, size_t>>
tryParseTable(std::span<Token> view) {
  std::span<Token> tab;
  Table table;
  size_t tokens = 0;

  std::optional<std::string> header = tryParseHeader(view);
  if (not header)
    return std::nullopt;
  table.setHeader(*header);
  tab = tab.subspan((*header).length());
  tokens += (*header).length();

  while (tab.size() > 0) {
    std::optional<std::pair<std::string, std::string>> kv =
        tryParseKeyValuePair(tab);
    if (kv) {
      table.addPair(*kv);
      tab = tab.subspan(3); // 2* string + Equal Token
      tokens += 3;
    } else {
      return std::make_pair<Table, size_t>(table, tokens);
    }
  }

  return std::nullopt;
}

std::optional<Toml> tryParse(TokenStream ts) {
  Toml toml;
  std::span<Token> view = ts.getViewAt(0);

  while (view.size() > 1) {
    std::optional<std::string> header = tryParseHeader(view);
    if (header) {
      std::optional<std::pair<Table, size_t>> table = tryParseTable(view);
      if (table) {
        std::pair<Table, size_t> tab = *table;
        toml.addTable(std::get<0>(tab));
        view = view.subspan(std::get<1>(tab));
      } else {
        // consume
      }
    } else {
      std::optional<std::pair<std::string, std::string>> pair =
          tryParseKeyValuePair(view);
      if (pair) {
        view = view.subspan(3); // 2*string + Equal Token
        toml.addKeyValuePair(*pair);
      } else {
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::toml

// https://toml.io/en/v1.0.0
