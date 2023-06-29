#include "ADT/Utf8String.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Lexer.h"
#include "Lexer/Token.h"
#include "Lexer/TokenStream.h"
#include "Location.h"

#include <cstdlib>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <string_view>
#include <system_error>
#include <unicode/uchar.h>
#include <unicode/ustdio.h>

using namespace rust_compiler::adt;

namespace rust_compiler::lexer {
/// https://doc.rust-lang.org/reference/crates-and-source-files.html
/// https://doc.rust-lang.org/stable/nightly-rustc/src/rustc_lexer/lib.rs.html
Token Lexer::advanceToken() {
  UChar32 next;

  if (auto c = bump())
    next = *c;
  else {
    return Token(getLocation(), TokenKind::Eof);
  }

  // check ASCII

  if (next == '/' && peek(1) == '/')
    lineComment();
  if (next == '/' && peek(1) == '*')
    blockComment();

  if (isWhiteSpace(next))
    skipWhiteSpace();

  if (next == '_' && isIdStart(1))
    return lexIdentifierOrKeyWord();
  if (isIdStart(0))
    return lexIdentifierOrKeyWord();

  switch (next) {

  // raw identifier, raw string literal, or identifier
  case 'r': {
    if (peek(1) == '#' && isIdStart(2))
      return lexRawIdentifier();
    if (peek(1) == '#' || peek(1) == '\"')
      return lexRawDoubleQuotedString();
    return lexIdentifierOrUnknownPrefix();
    break;
  }

    // byte literal, byte string literal, raw byte string literal, or identifier
  case 'b': {
    if (peek(1) == '\'')
      return lexByte();
    if (peek(1) == '\"')
      return lexByteString();
    if (peek(1) == 'r' && peek(2) == '"')
      return lexRawDoubleQuotedString();
    if (peek(1) == 'r' && peek(2) == '#')
      return lexRawDoubleQuotedString();
    return lexIdentifierOrUnknownPrefix();
  }

    // acount for BOM?

    // Identifier

  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return lexNumericalLiteral();

    // one-symbol tokens
  case ';':
    return Token(getLocation(), TokenKind::Semi);
  case ',':
    return Token(getLocation(), TokenKind::Comma);
  case '.':
    return Token(getLocation(), TokenKind::Dot);
  case '(':
    return Token(getLocation(), TokenKind::ParenOpen);
  case ')':
    return Token(getLocation(), TokenKind::ParenClose);
  case '{':
    return Token(getLocation(), TokenKind::BraceOpen);
  case '}':
    return Token(getLocation(), TokenKind::BraceClose);
  case '[':
    return Token(getLocation(), TokenKind::SquareOpen);
  case ']':
    return Token(getLocation(), TokenKind::SquareClose);
  case '@':
    return Token(getLocation(), TokenKind::At);
  case '#': {
    return Token(getLocation(), TokenKind::Hash);
  case '~':
    return Token(getLocation(), TokenKind::Tilde);
  case '?':
    return Token(getLocation(), TokenKind::QMark);
  case ':':
    return Token(getLocation(), TokenKind::Colon);
  case '$':
    return Token(getLocation(), TokenKind::Dollar);
  case '=':
    return Token(getLocation(), TokenKind::Eq);
  case '!':
    return Token(getLocation(), TokenKind::Not);
  case '<':
    return Token(getLocation(), TokenKind::Lt);
  case '>':
    return Token(getLocation(), TokenKind::Gt);
  case '-':
    return Token(getLocation(), TokenKind::Minus);
  case '&':
    return Token(getLocation(), TokenKind::And);
  case '|':
    return Token(getLocation(), TokenKind::Or);
  case '+':
    return Token(getLocation(), TokenKind::Plus);
  case '*':
    return Token(getLocation(), TokenKind::Star);
  case '^':
    return Token(getLocation(), TokenKind::Caret);
  case '%':
    return Token(getLocation(), TokenKind::Percent);

  // Lifetime or character literal
  case '\'':
    return lexLifetimeOrChar();

  // string literal
  case '\"':
    return lexStringLiteral();

  // Identifier starting with an emoji.
  default: {
    if (!isASCII() && isEmoji())
      return lexFakeIdentifierOrUnknownPrefix();
    // Error report
    llvm::errs() << "unknown token: " << next << "\n";
    assert(false);
  }
  }
  }
  llvm_unreachable("unknown token");
}

bool Lexer::isIdStart(int i) {
  return u_hasBinaryProperty(getUchar(i), UCHAR_XID_START);
}

bool Lexer::isIdContinue(int i) {
  return u_hasBinaryProperty(getUchar(i), UCHAR_XID_CONTINUE);
}

UChar32 Lexer::getUchar(int i) { return peek(i); }

Token Lexer::lexNumericalLiteral() {
  if (peek() == 'b')
    return lexBinLiteral();
  else if (peek() == 'o')
    return lexOctLiteral();
  else if (peek() == 'x')
    return lexHexLiteral();

  return lexDecOrFloatLiteral();
}

Token Lexer::lexDecOrFloatLiteral() {
  CheckPoint checkpoint = getCheckPoint();

  UChar32 current = getUchar();
  while (u_isdigit(current) or current == '_') {
    skip();
    current = getUchar();
  }
  recover(checkpoint);
  if (current == '.')
    return lexFloatLiteral();
  return lexDecimalLiteral();
}

Token Lexer::lexFloatLiteral() {
  std::string storage;

  // lex decimal literal
  UChar32 current = getUchar();
  while (u_isdigit(current) or current == '_') {
    storage += current;
    skip();
    current = getUchar();
  }

  if (current == '.') {
    storage += '.';
    // after dot
    while (u_isdigit(current) or current == '_') {
      storage += current;
      skip();
      current = getUchar();
    }
  } else if (current == 'e' or current == 'E') { // exponent
    storage += current;
    skip();
    current = getUchar();
    if (current == '+' or current == '-') {
      storage += current;
      skip();
      current = getUchar();
    }
  } else {
    // report error
  }

  if (checkFloatTypeHint()) {
  }
  assert(false);
}

// Token Lexer::lexRawIdentifier() {}

Token Lexer::lexIdentifierOrKeyWord() {
  Location loc = getLocation();

  UChar32 current = getUchar();

  adt::Utf8String identifier;

  // location!!!!!
  if (current == '_') {
    identifier.append(current);
    skip();

    current = getUchar();

    // _ XID_Continue +
    if (!u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
      // report error
    }

    identifier.append(current);
    skip();
    current = getUchar();

    while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
      identifier.append(current);
      skip();
      current = getUchar();
    } // report EOF!!!

    if (identifier.isASCII()) {
      if (auto keyword = isKeyWord(identifier.toString()))
        return Token(loc, *keyword, identifier.toString());
    }

    return Token(loc, TokenKind::Identifier, identifier);
  } else if (isIdStart()) {
    identifier.append(current);
    skip();

    current = getUchar();
    while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
      identifier.append(current);
      skip();
      current = getUchar();
    } // report EOF!!!

    if (identifier.isASCII()) {
      if (auto keyword = isKeyWord(identifier.toString()))
        return Token(loc, *keyword, identifier.toString());
    }

    return Token(loc, TokenKind::Identifier, identifier);
  }
  llvm::errs() << getLocation().toString()
               << ": failed to lex identifier: wrong prefix"
               << "\n";
  exit(EXIT_FAILURE);
}

void Lexer::lex(std::string_view fileName) {
  UFILE *file = u_fopen(fileName.data(), "r", nullptr, nullptr);

  UChar32 token = u_fgetcx(file);
  tokens.push_back(token);

  while (token != U_EOF) {
    token = u_fgetcx(file);
    tokens.push_back(token);
  }

  u_fclose(file);

  offset = 0;
}

bool Lexer::isWhiteSpace(UChar32 next) {
  return u_hasBinaryProperty(next, UCHAR_PATTERN_WHITE_SPACE);
}

void Lexer::skipWhiteSpace() {
  UChar32 current = getUchar();

  while (isWhiteSpace(current)) {
    skip();
    current = getUchar();
  }
}

UChar32 Lexer::peek(int i) {
  if (offset + i < tokens.size()) {
    return tokens[offset + i];
  } else {
    llvm::errs() << "peek beyond end of tokens"
                 << "\n";
    exit(EXIT_FAILURE);
  }
}

Location Lexer::getLocation() {
  return Location(fileName, lineNumber, columnNumber);
}

void Lexer::skip() {
  UChar32 current = getUchar();

  if (u_hasBinaryProperty(current, UCHAR_LINE_BREAK)) {
    ++offset;
    ++lineNumber;
    columnNumber = 0;
  } else {
    ++offset;
    ++columnNumber;
  }
}

bool Lexer::isASCII() { return getUchar() < 128; }

Token Lexer::lexByte() {
  adt::Utf8String byte;
  Location loc = getLocation();

  UChar b = getUchar();
  if (b != 0062) { // a b
    llvm::errs() << "failed to lex byte"
                 << "\n";
    exit(EXIT_FAILURE);
  }
  byte.append(b);
  skip();
  b = getUchar();
  if (b != 0027) { // a single-quote
    llvm::errs() << "failed to lex byte"
                 << "\n";
    exit(EXIT_FAILURE);
  }
  byte.append(b);
  skip();
  b = getUchar();
  if (b == '\\') { // a slash: BYTE_ESCAPE
    byte.append(b);
    skip();
    b = getUchar();
    if (b == 'n' || b == 'r' || b == 'r' || b == '\\' || b == '\'' ||
        b == '\"') { // not a slash
      byte.append(b);
      skip();
      b = getUchar();
    } else if (b == 'x') { // HEX_DIGIT  HEX_DIGIT
      byte.append(b);
      skip();
      b = getUchar();
      if (!u_hasBinaryProperty(b, UCHAR_ASCII_HEX_DIGIT)) {
        llvm::errs() << "failed to lex byte"
                     << "\n";
        exit(EXIT_FAILURE);
      }
      byte.append(b);
      skip();
      b = getUchar();
      if (!u_hasBinaryProperty(b, UCHAR_ASCII_HEX_DIGIT)) {
        llvm::errs() << "failed to lex byte"
                     << "\n";
        exit(EXIT_FAILURE);
      }
      byte.append(b);
      skip();
      b = getUchar();
    } else {
      llvm::errs() << "failed to lex byte"
                   << "\n";
      exit(EXIT_FAILURE);
    }
  } else if (b <= 0x7F && b != '\'' && b != '\\' && b != '\n' && b != '\r' &&
             b != '\t') { // not a slash: ASCII_FOR_CHAR
    byte.append(b);
    skip();
    b = getUchar();
  } else {
    llvm::errs() << "failed to lex byte"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  if (b != 0027) { // a single-quote
    llvm::errs() << "failed to lex byte"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  byte.append(b); // the terminating '
  skip();

  // FIXME SUFFIX?

  return Token(loc, TokenKind::BYTE_LITERAL, byte);
}

void Lexer::lineComment() {
  UChar32 c = getUchar();
  if (c != '/') {
    llvm::errs() << getLocation().toString() << ": failed to lex line comment"
                 << "\n";
    exit(EXIT_FAILURE);
  }
  skip();
  c = getUchar();
  if (c != '/') {
    llvm::errs() << getLocation().toString() << ": failed to lex line comment"
                 << "\n";
    exit(EXIT_FAILURE);
  }
  skip();

  // found //
  c = getUchar();
  while (!u_hasBinaryProperty(c, UCHAR_LINE_BREAK)) {
    skip();
    c = getUchar();
  }
}

std::optional<UChar32> Lexer::bump() {
  if (offset + 1 < tokens.size()) {
    ++offset;
    return tokens[offset];
  }
  return std::nullopt;
}

Token Lexer::lexIntegerLiteral() {
  UChar32 t = getUchar();
  if ('0' <= t && t <= '9')
    return lexDecimalLiteral();
  if (t == '0' && peek(1) == 'b')
    return lexBinLiteral();
  if (t == '0' && peek(1) == 'o')
    return lexOctLiteral();
  if (t == '0' && peek(1) == 'x')
    return lexHexLiteral();

  llvm::errs() << getLocation().toString() << ": failed to lex integer literal"
               << "\n";
  exit(EXIT_FAILURE);
}

Token Lexer::lexDecimalLiteral() {
  std::string literal;
  Location loc = getLocation();

  UChar32 current = getUchar();
  while (u_isdigit(current) or current == '_') {
    literal += current;
    skip();
    current = getUchar();
  }

  if (checkIntegerTypeHint())
    return Token(loc, TokenKind::INTEGER_LITERAL, literal, getTypeHint());

  return Token(loc, TokenKind::INTEGER_LITERAL, literal);
}

Token Lexer::lexBinLiteral() {
  std::string literal;
  Location loc = getLocation();

  UChar32 current = getUchar();

  if (current == '0' && peek(1) == 'b') {
    skip(); // '0'
    skip(); // 'b'
  } else {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex bin literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  current = getUchar();

  size_t binDigits = 0;
  while (current == '0' or current == '1' or current == '_') {
    if (current == '0' or current == '1')
      ++binDigits;
    literal += current;
    skip();
    current = getUchar();
  }

  if (binDigits == 0)
    llvm::errs() << getLocation().toString()
                 << ": failed to lex bin literal: no bin digits"
                 << "\n";
  exit(EXIT_FAILURE);

  if (checkIntegerTypeHint())
    return Token(loc, TokenKind::INTEGER_LITERAL, literal, getTypeHint());

  return Token(loc, TokenKind::INTEGER_LITERAL, literal);
}

Token Lexer::lexOctLiteral() {
  std::string literal;
  Location loc = getLocation();

  UChar32 current = getUchar();

  if (current == '0' && peek(1) == 'o') {
    skip(); // '0'
    skip(); // 'o'
  } else {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex bin literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  current = getUchar();
  size_t octDigits = 0;
  while ((current >= '0' and current <= '7') or current == '_') {
    if (current >= '0' and current <= '7')
      ++octDigits;
    literal += current;
    skip();
    current = getUchar();
  }

  if (octDigits == 0)
    llvm::errs() << getLocation().toString()
                 << ": failed to lex bin literal: no oct digits"
                 << "\n";
  exit(EXIT_FAILURE);

  if (checkIntegerTypeHint())
    return Token(loc, TokenKind::INTEGER_LITERAL, literal, getTypeHint());

  return Token(loc, TokenKind::INTEGER_LITERAL, literal);
}

Token Lexer::lexHexLiteral() {
  std::string literal;
  Location loc = getLocation();

  UChar32 current = getUchar();

  if (current == '0' && peek(1) == 'x') {
    skip(); // '0'
    skip(); // 'x'
  } else {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex hex literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  current = getUchar();
  size_t hexDigits = 0;
  while (u_hasBinaryProperty(current, UCHAR_ASCII_HEX_DIGIT) or
         current == '_') {
    if (u_hasBinaryProperty(current, UCHAR_ASCII_HEX_DIGIT))
      ++hexDigits;
    literal += current;
    skip();
    current = getUchar();
  }

  if (hexDigits == 0)
    llvm::errs() << getLocation().toString()
                 << ": failed to lex hex literal: no hex digits"
                 << "\n";
  exit(EXIT_FAILURE);

  if (checkIntegerTypeHint())
    return Token(loc, TokenKind::INTEGER_LITERAL, literal, getTypeHint());

  return Token(loc, TokenKind::INTEGER_LITERAL, literal);
}

Token Lexer::lexByteString() {
  Location loc = getLocation();

  UChar32 current = getUchar();
  Utf8String literal;

  if (current == 'b' && peek(1) == '"') {
    skip(); // 'b'
    literal.append('b');
    skip(); // '"'
    literal.append('"');
  } else {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex byte string literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  bool byteStringContinue = true;
  while (byteStringContinue) {
    if (isASCIIForString(current, peek(1))) {
      literal.append(current);
      skip();
      current = getUchar();
      byteStringContinue = true;
    } else if (isByteEscape()) {
      byteStringContinue = true;
      current = getUchar();
    } else if (isStringContinue()) {
      byteStringContinue = true;
      current = getUchar();
    } else if (current == '"') {
      literal.append('"');
      skip();
      byteStringContinue = false;
      current = getUchar();
    } else {
      llvm::errs() << getLocation().toString()
                   << ": failed to lex byte string literal: wrong postfix"
                   << "\n";
      exit(EXIT_FAILURE);
    }
  }

  if (checkSuffix()) {
    adt::Utf8String suffix = lexSuffixToUtf8();

    literal += suffix;
    return Token(loc, TokenKind::BYTE_STRING_LITERAL, literal);
  }

  return Token(loc, TokenKind::BYTE_STRING_LITERAL, literal);
}

bool Lexer::isASCIIForString(UChar32 current, UChar32 next) {
  return current <= 0x7F and current != '"' and current != '\\' and
         !(current == '\r' and next != '\n');
}

bool Lexer::isIsolatedCR() { return getUchar() == '\r' and peek(1) != '\n'; }

bool Lexer::checkSuffix() {
  UChar32 current = getUchar();

  if (isIdStart() || (current == '_' && isIdContinue(1)))
    return true;

  return false;
}

Token Lexer::lexStringLiteral() {
  Location loc = getLocation();

  adt::Utf8String literal;

  UChar32 current = getUchar();

  if (current != '"') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex string literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  skip(); // "
  literal.append('"');
  current = getUchar();

  while (true) {
    current = getUchar();
    if (current == '"') {
      // done
      literal.append('"');
      skip();
      if (checkSuffix()) {
        adt::Utf8String suffix = lexSuffixToUtf8();

        literal += suffix;
      }
      return Token(loc, TokenKind::STRING_LITERAL, literal);
    } else if (current == '\\' && (peek(1) == '\\' || peek(1) == '"')) {
      // escape char;
      literal.append('\\');
      skip();
      current = getUchar();
    } else {
      literal.append(current);
      skip();
      current = getUchar();
    }
  }
}

bool Lexer::isStringContinue() {
  UChar32 current = getUchar();

  return current == '\\' && peek(1) == '\n';
}

Token Lexer::lexLifetimeToken() {
  Location loc = getLocation();

  UChar32 current = getUchar();
  Utf8String storage;

  if (current != '\'') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex lifetime token: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  if (current == '\'' && peek(1) == '_') {
    storage.append('\'');
    storage.append('_');
    skip();
    skip();

    return Token(loc, TokenKind::LIFETIME_TOKEN, storage);
  }

  storage.append('\'');
  skip();

  Utf8String label = getIdentifierOrKeyWord();
  skipN(label.getLength());

  storage += label;
  return Token(loc, TokenKind::LIFETIME_TOKEN, storage);
}

Token Lexer::lexLifetimeOrLabel() {
  Location loc = getLocation();

  UChar32 current = getUchar();
  Utf8String storage;

  if (current != '\'') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex lifetime or label: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  storage.append('\'');
  skip();

  Utf8String label = getNonKeyWordIdentifier();
  skipN(label.getLength());

  storage += label;
  return Token(loc, TokenKind::LIFETIME_OR_LABEL, storage);
}

void Lexer::skipN(unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    skip();
}

bool Lexer::isASCIIEscape() {
  UChar32 current = getUchar();

  if (current == '\n')
    return true;
  if (current == '\r')
    return true;
  if (current == '\t')
    return true;
  if (current == '\\')
    return true;
  if (current == '\0')
    return true;

  if (current <= 0x7F)
    return true;

  return false;
}

CheckPoint Lexer::getCheckPoint() const {
  return CheckPoint(offset, lineNumber, columnNumber);
}

void Lexer::recover(const CheckPoint &cp) {
  offset = cp.getOffset();
  lineNumber = cp.getLineNumber();
  columnNumber = cp.getColumnNumber();
}

bool Lexer::isQuoteEscape() {
  UChar32 current = getUchar();

  return current == '\\' && (peek(1) == '\'' || peek(1) == '"');
}

Token Lexer::lexRawIdentifier() {
  Location loc;

  UChar32 current = getUchar();
  if (current != 'r' || peek(1) != '#') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex raw identifier: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  skip(); // 'r'
  skip(); // '#'

  current = getUchar();

  Utf8String storage;

  if (isIdStart()) {
    storage.append(current);
    skip();
    while (isIdContinue()) {
      storage.append(current);
      skip();
    }
    if (storage.isEqualASCII("crate") || storage.isEqualASCII("self") ||
        storage.isEqualASCII("super") || storage.isEqualASCII("Self")) {
      llvm::errs() << getLocation().toString()
                   << ": failed to lex raw identifier: found crate, self, "
                      "super, or Self"
                   << "\n";
      exit(EXIT_FAILURE);
    }
    return Token(loc, TokenKind::Identifier, storage);
  } else if (current == '_') {
    storage.append(current);
    skip();
    while (isIdContinue()) {
      storage.append(current);
      skip();
    }
    if (storage.isEqualASCII("crate") || storage.isEqualASCII("self") ||
        storage.isEqualASCII("super") || storage.isEqualASCII("Self")) {
      llvm::errs() << getLocation().toString()
                   << ": failed to lex raw identifier: found crate, self, "
                      "super, or Self"
                   << "\n";
      exit(EXIT_FAILURE);
    }
    return Token(loc, TokenKind::Identifier, storage);
  }

  llvm::errs()
      << getLocation().toString()
      << ": failed to lex raw identifier: found crate, self, super, or Self"
      << "\n";
  exit(EXIT_FAILURE);
}

bool Lexer::isByteEscape() {
  CheckPoint cp = getCheckPoint();

  UChar32 current = getUchar();

  if (current != '\\') {
    recover(cp);
    return false;
  }

  skip(); //
  current = getUchar();

  if (current == 'n' || current == 'r' || current == 't' || current == '\\' ||
      current == '0' || current == '\'' || current == '"') {
    recover(cp);
    return true;
  }

  if (current == 'x') {
    skip(); // x
    current = getUchar();

    if (u_hasBinaryProperty(current, UCHAR_ASCII_HEX_DIGIT) &&
        u_hasBinaryProperty(peek(1), UCHAR_ASCII_HEX_DIGIT)) {
      recover(cp);
      return true;
    }
  }
  recover(cp);
  return false;
}

Utf8String Lexer::lexSuffixToUtf8() { return getIdentifierOrKeyWord(); }

Utf8String Lexer::getIdentifierOrKeyWord() {
  Utf8String literal;
  UChar32 current = getUchar();

  if (current == '_' && u_hasBinaryProperty(peek(1), UCHAR_XID_CONTINUE)) {
    literal.append(current);
    skip();
    current = getUchar();

    while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
      literal.append(current);
      skip();
      current = getUchar();
    }
    return literal;

  } else if (u_hasBinaryProperty(current, UCHAR_XID_START)) {
    literal.append(current);
    skip();
    current = getUchar();
    while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
      literal.append(current);
      skip();
      current = getUchar();
    }
    return literal;
  }

  llvm::errs() << getLocation().toString()
               << ": failed to lex  identifier or keyword: wrong prefix"
               << "\n";
  exit(EXIT_FAILURE);
}

// Token Lexer::lexLifetimeOrLabel() {
//   Location loc = getLocation();
//   Utf8String literal;
//
//   UChar32 current = getUchar();
//
//   if (current != '\'') {
//     llvm::errs() << getLocation().toString()
//                  << ": failed to lex  lifetome or label: wrong prefix"
//                  << "\n";
//     exit(EXIT_FAILURE);
//   }
//
//   literal.append('\'');
//   skip();
//
//   Utf8String label = getIdentifierOrKeyWord();
//   if (!label.isASCII()) {
//     llvm::errs() << getLocation().toString()
//                  << ": failed to lex  lifetime or label: label is not
//                  ASCII"
//                  << "\n";
//     exit(EXIT_FAILURE);
//   }
//
//   std::string labelAsASCII = label.toString();
//   if (isKeyWord(labelAsASCII)) {
//     llvm::errs() << getLocation().toString()
//                  << ": failed to lex  lifetime or label: label is a
//                  keyword"
//                  << "\n";
//     exit(EXIT_FAILURE);
//   }
//
//   literal += label;
//
//   return Token(loc, TokenKind::LIFETIME_OR_LABEL, literal);
// }

Token Lexer::lexLifetimeOrChar() {
  CheckPoint cp = getCheckPoint();
  UChar32 current = getUchar();

  if (current != '\'') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex  lifetime or char"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  bool canBeALifetime =
      (peek(2) == '\'') ? false : isIdStart(1) || u_isdigit(peek(1));

  if (!canBeALifetime)
    return lexChar();

  skip(); // \'
  current = getUchar();

  while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
    skip();
    current = getUchar();
  }

  if (current == '\'') {
    skip();
    recover(cp);
    return lexChar();
  }
  recover(cp);
  return lexLifetime();
}

Token Lexer::lexLifetime() {
  CheckPoint cp = getCheckPoint();
  UChar32 current = getUchar();

  if (current != '\'') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex  lifetime: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  skip();

  current = getUchar();

  if (current == '_') {
    recover(cp);
    return lexLifetimeToken();
  }

  Utf8String label = getIdentifierOrKeyWord();
  if (label.isASCII()) {
    std::string labelAsASCII = label.toString();
    if (isKeyWord(labelAsASCII)) {
      recover(cp);
      return lexLifetimeToken();
    }
  }
  recover(cp);
  return lexLifetimeOrLabel();
}

Token Lexer::lexRawDoubleQuotedString() {
  Location loc = getLocation();
  Utf8String storage;
  UChar32 current = getUchar();

  if (current != 'b' || peek(1) != 'r') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex  raw byte string literal: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  // storage.append('b');
  // storage.append('r');
  skip();
  skip();

  current = getUchar();

  Utf8String rawString = getRawByteStringContent();

  llvm::errs() << getLocation().toString()
               << ": failed to lex  raw byte string literal: wrong prefix"
               << "\n";
  exit(EXIT_FAILURE);
}

adt::Utf8String Lexer::getRawByteStringContent() {
  Utf8String storage;
  UChar32 current = getUchar();
  if (current == '"') {
    skip();
    current = getUchar();
    while (current < 128 && current != '"') {
      storage.append(current);
      skip();
      current = getUchar();
    }

    if (current == '"') {
      skip();
      return storage;
    }
    llvm::errs() << getLocation().toString()
                 << ": failed to lex  raw byte string content: wrong postfix"
                 << "\n";
    exit(EXIT_FAILURE);

  } else if (current == '#') {
    skip();
    storage = getRawByteStringContent();
    current = getUchar();
    if (current != '#') {
      llvm::errs() << getLocation().toString()
                   << ": failed to lex  raw byte string content: wrong postfix"
                   << "\n";
      exit(EXIT_FAILURE);
    }
    return storage;
  }
  llvm::errs() << getLocation().toString()
               << ": failed to lex  raw byte string content: wrong prefix"
               << "\n";
  exit(EXIT_FAILURE);
}

Token Lexer::lexChar() {
  Location loc = getLocation();
  Utf8String literal;

  UChar32 current = getUchar();

  if (current == '\'' && peek(1) != '\\' && peek(2) == '\'') {
    literal.append(current);
    literal.append(peek(1));
    literal.append(peek(2));

    return Token(loc, TokenKind::CHAR_LITERAL, literal);
  }
  llvm::errs() << getLocation().toString()
               << ": failed to lex char content: wrong prefix"
               << "\n";
  exit(EXIT_FAILURE);
}

bool Lexer::isEmoji() {
  return u_hasBinaryProperty(peek(0), UCHAR_EMOJI) &&
         u_hasBinaryProperty(peek(0), UCHAR_BASIC_EMOJI);
}

Token Lexer::lexIdentifierOrUnknownPrefix() {
  Location loc = getLocation();
  UChar32 current = getUchar();
  Utf8String literal;

  if (!u_hasBinaryProperty(current, UCHAR_XID_START) && current != '_') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex identifier or unknown prefix: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  literal.append(current);
  skip();
  current = getUchar();

  while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
    literal.append(current);
    skip();
    current = getUchar();
  }

  if (literal.isASCII()) {
    if (auto keyword = isKeyWord(literal.toString()))
      return Token(loc, *keyword, literal.toString());
  }

  return Token(loc, TokenKind::Identifier, literal);
}

Token Lexer::lexFakeIdentifierOrUnknownPrefix() {
  Location loc = getLocation();
  UChar32 current = getUchar();
  Utf8String literal;

  if (!u_hasBinaryProperty(current, UCHAR_XID_START) && current != '_') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex identifier or unknown prefix: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  literal.append(current);
  skip();
  current = getUchar();

  while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
    literal.append(current);
    skip();
    current = getUchar();
  }

  if (literal.isASCII()) {
    if (auto keyword = isKeyWord(literal.toString()))
      return Token(loc, *keyword, literal.toString());
  }

  return Token(loc, TokenKind::Identifier, literal);
}

void Lexer::blockComment() {
  Location loc = getLocation();
  UChar32 current = getUchar();

  if (current != '/') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex block comment: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  skip();
  current = getUchar();

  if (current != '*') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex block comment: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  skip();
  current = getUchar();

  size_t depth = 1;
  while (true) {
    if (current == '/' && peek(1) == '*') {
      skip();
      current = getUchar();
      ++depth;
    } else if (current == '*' && peek(1) == '/') {
      skip();
      current = getUchar();
      --depth;
      if (depth == 0)
        break;
    }
  }
}

bool Lexer::checkIntegerTypeHint() {
  UChar32 current = getUchar();

  if (current == 'i' || current == 'u') {
    if (peek(1) == '8')
      return true;

    if (peek(1) == '1' && peek(2) == '6')
      return true;

    if (peek(1) == '3' && peek(2) == '2')
      return true;

    if (peek(1) == '6' && peek(2) == '4')
      return true;

    if (peek(1) == '1' && peek(2) == '2' && peek(3) == '8')
      return true;
  }

  return false;
}

bool Lexer::checkFloatTypeHint() {
  UChar32 current = getUchar();

  if (current == 'f' && peek(1) == '3' && peek(2) == '2')
    return true;

  if (current == 'f' && peek(1) == '6' && peek(2) == '4')
    return true;

  return false;
}

TypeHint Lexer::getTypeHint() {
  UChar32 current = getUchar();

  if (current == 'i') {
    if (peek(1) == '8')
      return TypeHint::i8;

    if (peek(1) == '1' && peek(2) == '6')
      return TypeHint::i16;

    if (peek(1) == '3' && peek(2) == '2')
      return TypeHint::i32;

    if (peek(1) == '6' && peek(2) == '4')
      return TypeHint::i64;

    if (peek(1) == '1' && peek(2) == '2' && peek(3) == '8')
      return TypeHint::i128;
  }

  if (current == 'u') {
    if (peek(1) == '8')
      return TypeHint::u8;

    if (peek(1) == '1' && peek(2) == '6')
      return TypeHint::u16;

    if (peek(1) == '3' && peek(2) == '2')
      return TypeHint::u32;

    if (peek(1) == '6' && peek(2) == '4')
      return TypeHint::u64;

    if (peek(1) == '1' && peek(2) == '2' && peek(3) == '8')
      return TypeHint::u128;
  }

  if (current == 'f' && peek(1) == '3' && peek(2) == '2')
    return TypeHint::f32;

  if (current == 'f' && peek(1) == '6' && peek(2) == '4')
    return TypeHint::f64;

  llvm::errs() << getLocation().toString() << ": failed to lex type hint"
               << "\n";
  exit(EXIT_FAILURE);
}

adt::Utf8String Lexer::getNonKeyWordIdentifier() {
  Utf8String identifier;
  UChar32 current = getUchar();

  if (!u_hasBinaryProperty(current, UCHAR_XID_START) && current != '_') {
    llvm::errs() << getLocation().toString()
                 << ": failed to lex non keyword identifier: wrong prefix"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  identifier.append(current);
  skip();
  current = getUchar();

  while (u_hasBinaryProperty(current, UCHAR_XID_CONTINUE)) {
    identifier.append(current);
    skip();
    current = getUchar();
  }

  if (identifier.isASCII()) {
    if (auto key = isKeyWord(identifier.toString())) {
      llvm::errs()
          << getLocation().toString()
          << ": failed to lex non keyword identifier: it is a keyword: "
          << "\n";
      exit(EXIT_FAILURE);
    }
  }

  return identifier;
}

} // namespace rust_compiler::lexer

/*
  underscore token
  identifier token


 */
