#include "Lexer.h"

#include "TokenStream.h"

#include <optional>

namespace rust_compiler {

std::string tryLexComment(std::string_view code) {
  std::string_view view = code;
  std::string ws;

  while (view.size() > 0) {
    if (view.starts_with("/")) {
      view.remove_prefix(1);
      ws.push_back('/');
    } else if (isdigit(view[0])) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (isalpha(view[0])) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(" ")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("_")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("-")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("?")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("\\")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("*")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("[")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("]")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("!")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("+")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(":")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(".")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("(")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(")")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("{")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("}")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(",")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("=")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("\"")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(";")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(".")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("&")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("(")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(")")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("<")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with(">")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("\'")) {
      ws.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("->")) {
      ws.push_back(view[0]);
      ws.push_back(view[1]);
      view.remove_prefix(2);
    } else if (view.starts_with("\n")) {
      view.remove_prefix(1);
      ws.push_back('\n');
      return ws;
    }
  }

  return ws;
}

std::string tryLexWhiteSpace(std::string_view code) {
  std::string_view view = code;
  std::string ws;

  while (view.size() > 0) {
    if (view.starts_with(" ")) {
      view.remove_prefix(1);
      ws.push_back(' ');
    } else if (view.starts_with("\t")) {
      view.remove_prefix(1);
      ws.push_back('\t');
    } else {
      return ws;
    }
  }

  return ws;
}

std::optional<std::string> tryLexIdentifier(std::string_view code) {
  std::string_view view = code;
  std::string id;

  while (view.size() > 0) {
    if (isalpha(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '_') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (isdigit(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else {
      if (id.size() > 0)
        return id;
      else
        return std::nullopt;
    }
  }

  return std::nullopt;
}

std::optional<std::string> tryLexString(std::string_view code) {
  std::string_view view = code;
  std::string id;

  if (!view.starts_with("\"")) {
    return std::nullopt;
  }

  id.push_back(view[0]);
  view.remove_prefix(1); // "

  while (view.size() > 0) {
    if (isalpha(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '_') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '-') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '/') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '{') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '}') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == ':') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '.') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == ' ') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (isdigit(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("\"")) {
      id.push_back(view[0]);
      view.remove_prefix(1);
      if (id.size() > 0)
        return id;
      else
        return std::nullopt;
    } else {
      printf("unknown string token: x%s\n", code.data());
      return std::nullopt;
    }
  }

  return std::nullopt;
}

std::optional<std::string> tryLexChar(std::string_view code) {
  std::string_view view = code;
  std::string id;

  if (!view.starts_with("\'")) {
    return std::nullopt;
  }

  id.push_back(view[0]);
  view.remove_prefix(1); // '

  while (view.size() > 0) {
    if (isalpha(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '_') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '-') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == ' ') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '.') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view[0] == '/') {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (isdigit(view[0])) {
      id.push_back(view[0]);
      view.remove_prefix(1);
    } else if (view.starts_with("\'")) {
      id.push_back(view[0]);
      view.remove_prefix(1);
      if (id.size() > 0)
        return id;
      else
        return std::nullopt;
    } else {
      printf("unknown char token: x%s\n", code.data());
      return std::nullopt;
    }
  }

  return std::nullopt;
}


TokenStream lex(std::string_view _code) {
  TokenStream ts;
  std::string_view code = _code;

  while (code.size() > 0) {

    //    printf("code.size(): %lu\n", code.size());
    //    printf("code.size(): %s\n", code.data());

    std::string ws = tryLexWhiteSpace(code);
    code.remove_prefix(ws.size());

    if (code.starts_with("//")) {
      std::string comment = tryLexComment(code);
      code.remove_prefix(comment.size());
      continue;
    }

    std::optional<std::string> str = tryLexString(code);
    if (str) {
      ts.append(Token(TokenKind::String, *str));
      code.remove_prefix(str->size());
      continue;
    }

    std::optional<std::string> ch = tryLexChar(code);
    if (ch) {
      ts.append(Token(TokenKind::Char, *ch));
      code.remove_prefix(ch->size());
      continue;
    }

    std::optional<std::string> id = tryLexIdentifier(code);
    if (id) {
      ts.append(Token(TokenKind::Identifier, *id));
      code.remove_prefix(id->size());
      continue;
    }

    if (code.starts_with("!")) {
      ts.append(Token(TokenKind::Exclaim));
      code.remove_prefix(1);
    } else if (code.starts_with("->")) {
      ts.append(Token(TokenKind::ThinArrow));
      code.remove_prefix(2);
    } else if (code.starts_with("+")) {
      ts.append(Token(TokenKind::Plus));
      code.remove_prefix(1);
    } else if (code.starts_with(".")) {
      ts.append(Token(TokenKind::Dot));
      code.remove_prefix(1);
    } else if (code.starts_with("?")) {
      ts.append(Token(TokenKind::QMark));
      code.remove_prefix(1);
    } else if (code.starts_with("*")) {
      ts.append(Token(TokenKind::Star));
      code.remove_prefix(1);
    } else if (code.starts_with("=")) {
      ts.append(Token(TokenKind::Equals));
      code.remove_prefix(1);
    } else if (code.starts_with(":")) {
      ts.append(Token(TokenKind::Colon));
      code.remove_prefix(1);
    } else if (code.starts_with("-")) {
      ts.append(Token(TokenKind::Dash));
      code.remove_prefix(1);
    } else if (code.starts_with(">>")) {
      ts.append(Token(TokenKind::DoubleGreaterThan));
      code.remove_prefix(2);
    } else if (code.starts_with(">")) {
      ts.append(Token(TokenKind::GreaterThan));
      code.remove_prefix(1);
    } else if (code.starts_with("<")) {
      ts.append(Token(TokenKind::LessThan));
      code.remove_prefix(1);
    } else if (code.starts_with("::")) {
      ts.append(Token(TokenKind::DoubleColon));
      code.remove_prefix(2);
    } else if (code.starts_with("&")) {
      ts.append(Token(TokenKind::Amp));
      code.remove_prefix(1);
    } else if (code.starts_with("&&")) {
      ts.append(Token(TokenKind::DoubleAmp));
      code.remove_prefix(2);
    } else if (code.starts_with("#")) {
      ts.append(Token(TokenKind::Hash));
      code.remove_prefix(1);
    } else if (code.starts_with("{")) {
      ts.append(Token(TokenKind::BraceOpen));
      code.remove_prefix(1);
    } else if (code.starts_with("}")) {
      ts.append(Token(TokenKind::BraceClose));
      code.remove_prefix(1);
    } else if (code.starts_with("[")) {
      ts.append(Token(TokenKind::SquareOpen));
      code.remove_prefix(1);
    } else if (code.starts_with("]")) {
      ts.append(Token(TokenKind::SquareClose));
      code.remove_prefix(1);
    } else if (code.starts_with(",")) {
      ts.append(Token(TokenKind::Comma));
      code.remove_prefix(1);
    } else if (code.starts_with("(")) {
      ts.append(Token(TokenKind::ParenOpen));
      code.remove_prefix(1);
    } else if (code.starts_with(")")) {
      ts.append(Token(TokenKind::ParenClose));
      code.remove_prefix(1);
    } else if (code.starts_with("!")) {
      ts.append(Token(TokenKind::Exclaim));
      code.remove_prefix(1);
    } else if (code.starts_with(";")) {
      ts.append(Token(TokenKind::SemiColon));
      code.remove_prefix(1);
    } else if (code.starts_with("|")) {
      ts.append(Token(TokenKind::Pipe));
      code.remove_prefix(1);
    } else if (code.starts_with("\n")) {
      code.remove_prefix(1);
    } else {
      printf("unknown token: x%s\n", code.data());
      exit(EXIT_FAILURE);
    }
  }

  return ts;
}

} // namespace rust_compiler

// https://github.com/thepowersgang/mrustc/blob/master/src/parse/lex.cpp
