#include "Lexer/Lexer.h"
#include "Lexer/Token.h"

#include <llvm/Support/ErrorHandling.h>
#include <unicode/uchar.h>

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

/// https://stackoverflow.com/questions/1543613/how-does-utf-8-variable-width-encoding-work
UChar32 Lexer::getUchar(int i) {
  uint8_t input = peek(i);
  if (input < 128) {
    return input; // ASCII 1 byte
  } else if ((input & 0xC0) == 0x80) {
    // invalid
  } else if ((input & 0xE0) == 0xC0) {
    // 2 bytes
    uint8_t input2 = peek(i + 1);
    if ((input2 & 0xC0) != 0x80)
      return 0xFFFE;

    uint32_t output = ((input & 0x1F) << 6) | ((input2 & 0x3F) << 0);
    return output;
  } else if ((input & 0xF0) == 0xE0) {
    // 3 bytes
    uint8_t input2 = peek(i + 1);
    if ((input2 & 0xC0) != 0x80)
      return 0xFFFE;

    uint8_t input3 = peek(i + 2);
    if ((input3 & 0xC0) != 0x80)
      return 0xFFFE;

    uint32_t output = ((input & 0x0F) << 12) | ((input2 & 0x3F) << 6) |
                      ((input3 & 0x3F) << 0);
    return output;
  } else if ((input & 0xF8) == 0xF0) {
    // 4 bytes
    uint8_t input2 = peek(i + 1);
    if ((input2 & 0xC0) != 0x80)
      return 0xFFFE;

    uint8_t input3 = peek(i + 2);
    if ((input3 & 0xC0) != 0x80)
      return 0xFFFE;

    uint8_t input4 = peek(i + 3);
    if ((input4 & 0xC0) != 0x80)
      return 0xFFFE;

    uint32_t output = ((input & 0x07) << 18) | ((input2 & 0x3F) << 12) |
                      ((input3 & 0x3F) << 6) | ((input4 & 0x3F) << 0);
    return output;
  } else
    // report error
    llvm::errs() << getLocation().toString() << "invalid UTF-8"
                 << "\n";
  return 0xFFFE;
}

Token Lexer::lexNumericalLiteral() {
  if (peek() == 'b')
    return lexBinLiteral();
  else if (peek() == 'o')
    return lexOctLiteral();
  else if (peek() == 'x')
    return lexHexLiteral();

  return lexDecOrFloatLiteral();
}

// Token Lexer::lexRawIdentifier() {}

Token Lexer::lexIdentifierOrKeyWord() {
  if (peek() == '_') {
  } else {
    UChar32 next = getUchar();
    // XID_START
    if (!u_hasBinaryProperty(next, UCHAR_XID_START)) {
      // report error
    }
    //skip32(next);
  }

  //UChar32 next = getUchar();

  // XID_Continue*
  //UChar32 codepoint;
  assert(false);
}

} // namespace rust_compiler::lexer

/*
  underscore token
  identifier token


 */
