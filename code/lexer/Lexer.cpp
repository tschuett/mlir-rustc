#include "Lexer/Lexer.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"

#include <llvm/Support/raw_ostream.h>
#include <optional>

namespace rust_compiler::lexer {

using rust_compiler::Location;

static const std::pair<IntegerKind, std::string> IK[] = {
    {IntegerKind::I8, "i8"},       {IntegerKind::I16, "i16"},
    {IntegerKind::I32, "i32"},     {IntegerKind::I64, "i64"},
    {IntegerKind::I128, "i128"},   {IntegerKind::U8, "u8"},
    {IntegerKind::U16, "u16"},     {IntegerKind::U32, "u32"},
    {IntegerKind::U64, "u64"},     {IntegerKind::U128, "u128"},
    {IntegerKind::ISize, "isize"}, {IntegerKind::USize, "usize"},
};

static const std::pair<FloatKind, std::string> FK[] = {
    {FloatKind::F32, "f32"},
    {FloatKind::F64, "f64"},
};

std::optional<std::string> tryLexDecLiteral(std::string_view code) {
  std::string_view view = code;
  std::string ws = "";

  if (view.size() > 0) {
    // single digit
    if (std::isdigit(view.front()) && not std::isdigit(view[1])) {
      ws.push_back(view.front());
      return ws;
    }
    if (std::isdigit(view.front()) && std::isdigit(view[1]) &&
        not std::isdigit(view[2])) {
      ws.push_back(view.front());
      ws.push_back(view[1]);
      return ws;
    }

    if (std::isdigit(view.front()) && std::isdigit(view[1]) &&
        std::isdigit(view[2]) && not std::isdigit(view[3])) {
      ws.push_back(view.front());
      ws.push_back(view[1]);
      ws.push_back(view[2]);
      return ws;
    }

    while (view.size() > 0) {
      if (std::isdigit(view.front())) {
        ws.push_back(view.front());
        view.remove_prefix(1);
      } else {
        return std::nullopt;
      }
    }
    return ws;
  }
  return std::nullopt;
}

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
  std::string id = "";

  if (isdigit(view.front()))
    return std::nullopt;

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
      else {
        // printf("unknown identifier token: x%sx\n", code.data());
        return std::nullopt;
      }
    }
  }

  if (id.size() > 0)
    return id;
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

TokenStream lex(std::string_view _code, std::string_view fileName) {
  TokenStream ts;
  std::string_view code = _code;
  unsigned lineNumber = 0;
  unsigned columnNumber = 0;

  while (code.size() > 0) {

    // printf("code.size(): %lu\n", code.size());
    // printf("code.size(): %s\n", code.data());

    std::string ws = tryLexWhiteSpace(code);
    code.remove_prefix(ws.size());

    //    for (auto &ik : IK) {
    //      if (code.starts_with(std::get<1>(ik))) {
    //        ts.append(Token(Location(fileName, lineNumber, columnNumber),
    //                        std::get<0>(ik)));
    //        code.remove_prefix(std::get<1>(ik).size());
    //        columnNumber += std::get<1>(ik).size();
    //        continue;
    //      }
    //    }
    //
    //    for (auto &fk : FK) {
    //      if (code.starts_with(std::get<1>(fk))) {
    //        ts.append(Token(Location(fileName, lineNumber, columnNumber),
    //                        std::get<0>(fk)));
    //        code.remove_prefix(std::get<1>(fk).size());
    //        columnNumber += std::get<1>(fk).size();
    //        continue;
    //      }
    //    }

    ws = tryLexWhiteSpace(code);
    code.remove_prefix(ws.size());

    if (code.starts_with("//")) {
      std::string comment = tryLexComment(code);
      code.remove_prefix(comment.size());
      columnNumber += comment.size();
      continue;
    }

    std::optional<std::string> decInt = tryLexDecLiteral(code);
    if (decInt) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::INTEGER_LITERAL, *decInt));
      code.remove_prefix(decInt->size());
      columnNumber += decInt->size();
      continue;
    }

    std::optional<std::string> str = tryLexString(code);
    if (str) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::STRING_LITERAL, *str));
      code.remove_prefix(str->size());
      columnNumber += str->size();
      continue;
    }

    std::optional<std::string> ch = tryLexChar(code);
    if (ch) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Char, *ch));
      code.remove_prefix(ch->size());
      columnNumber += ch->size();
      continue;
    }

    std::optional<std::string> id = tryLexIdentifier(code);
    if (id) {
      if (isKeyWord(*id)) {
        ts.append(Token(Location(fileName, lineNumber, columnNumber),
                        *isKeyWord(*id), *id));
        code.remove_prefix(id->size());
        columnNumber += id->size();
        continue;
      } else {
        ts.append(Token(Location(fileName, lineNumber, columnNumber),
                        TokenKind::Identifier, *id));
        code.remove_prefix(id->size());
        columnNumber += id->size();
        continue;
      }
    }

    if (code.starts_with("!")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Not));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("$crate")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Identifier, "$crate"));
      code.remove_prefix(6);
      columnNumber += 6;
    } else if (code.starts_with("->")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::RArrow));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("+")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Plus));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(".")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Dot));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("?")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::QMark));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("*")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Star));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("==")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::EqEq));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("!=")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Ne));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("=")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Eq));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("-")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Minus));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(">>")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Shr));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with(">")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Gt));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("<<")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Shl));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("<=")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Le));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("::")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::PathSep));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with(":")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Colon));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("&&")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::AndAnd));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("&")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::And));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("#")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Hash));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("{")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::BraceOpen));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("}")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::BraceClose));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("[")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::SquareOpen));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("]")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::SquareClose));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(",")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Comma));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("(")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::ParenOpen));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(")")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::ParenClose));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("!")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Not));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(";")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Semi));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("|")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Or));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("!")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Not));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("^")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Caret));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("%")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Percent));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("/")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::Slash));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with(">=")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Ge));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("<=")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Le));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("<")) {
      ts.append(
          Token(Location(fileName, lineNumber, columnNumber), TokenKind::Lt));
      code.remove_prefix(1);
      columnNumber += 1;
    } else if (code.starts_with("..")) {
      ts.append(Token(Location(fileName, lineNumber, columnNumber),
                      TokenKind::DotDot));
      code.remove_prefix(2);
      columnNumber += 2;
    } else if (code.starts_with("\n")) {
      code.remove_prefix(1);
      ++lineNumber;
      columnNumber = 0;
    } else {
      if (code.size() == 0) {
        ts.append(Token(Location(fileName, lineNumber, columnNumber),
                        TokenKind::Eof));
        return ts;
      }
      printf("unknown token: x%sx\n", code.data());
      llvm::outs() << "remaining chars:" << code.size() << "\n";
      ts.print(50);
      exit(EXIT_FAILURE);
    }
  }
  llvm::outs() << "lexer: done"
               << "\n";

  ts.append(
      Token(Location(fileName, lineNumber, columnNumber), TokenKind::Eof));

  // ts.print(50);
  return ts;
}

} // namespace rust_compiler::lexer

// https://github.com/thepowersgang/mrustc/blob/master/src/parse/lex.cpp

// https://doc.rust-lang.org/reference/tokens.html#punctuation
