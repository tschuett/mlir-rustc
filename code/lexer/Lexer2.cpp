#include "Lexer/Lexer.h"
#include "Lexer/Token.h"
#include "llvm/Support/ErrorHandling.h"
#include "unicode/uchar.h"

namespace rust_compiler::lexer {

/// https://doc.rust-lang.org/stable/nightly-rustc/src/rustc_lexer/lib.rs.html
Token Lexer::advanceToken() {
  char next;

  if (auto c = bump())
    next = *c;
  else {
    return Token(getLocation(), TokenKind::Eof);
  }

  if (next == '/' && peek() == '/')
    lineComment();
  if (next == '/' && peek() == '*')
    blockComment();

  if (isWhiteSpace())
    skipWhiteSpace();

  switch (next) {

  case 'r': {
    if (peek() == '#' && isIdStart(1))
      return lexRawIdentifier();
    break;
  }

  case 'b': {
    if (peek() == '\'')
      return lexByte();
    if (peek() == '\"')
      return lexByteString();
    if (peek() == 'r' && peek(1) == '"')
      return lexRawDoubleQuotedString();
    if (peek() == 'r' && peek(1) == '#')
      return lexRawDoubleQuotedString();
    return lexIdentifierOrUnknownPrefix();
  }

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
  case '\'':
    return lexLifetimeOrChar();
  case '\"':
    return lexStringLiteral();
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

UChar32 Lexer::getUchar(int i) {
  uint8_t input = peek(i);
  if (input < 128)
    return input; // ASCII 1 byte
  // FIXME
  assert(false);
}

Token Lexer::lexNumericalLiteral() {
  if (peek() == 'b')
    lexBinLiteral();
  else if (peek() == 'o')
    lexOctLiteral();
  else if (peek() == 'x')
    lexHexLiteral();

  return lexDecOrFloatLiteral();
}

} // namespace rust_compiler::lexer

/*
  underscore token
  identifier token


 */
