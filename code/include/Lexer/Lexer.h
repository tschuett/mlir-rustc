#pragma once

#include "Lexer/TokenStream.h"

#include <string_view>
#include <unicode/uchar.h>

/// Rust input is interpreted as a sequence of Unicode code points encoded in
/// UTF-8.
namespace rust_compiler::lexer {

TokenStream lex(std::string_view code, std::string_view fileName);

class CheckPoint {
  uint32_t offset;
  uint32_t lineNumber;
  uint32_t columnNumber;
public:

  uint32_t getOffset() const { return offset;}
  uint32_t getLineNumber() const { return lineNumber;}
  uint32_t getColumnNumber() const { return columnNumber;}
};

/// https://doc.rust-lang.org/reference/tokens.html
class Lexer {
  // std::string chars;
  std::string fileName;
  uint32_t remaining;
  TokenStream tokenStream;
  uint32_t offset;

  std::vector<UChar32> tokens;

public:
  void lex(std::string_view fileName);

private:
  std::optional<UChar32> bump();

  Token advanceToken();

  Token lexChar();
  Token lexString();
  Token lexRawString();
  Token lexByte();
  Token lexByteString();
  Token lexRawByte();
  Token lexRawByteString();
  Token lexStringLiteral();
  Token lexRawDoubleQuotedString();
  Token lexRawIdentifier();

  Token lexIntegerLiteral();
  Token lexFloatLiteral();

  Token lexLifetimeToken();
  Token lexLifetimeOrLabel();
  Token lexLifetimeOrChar();

  Token lexNumericalLiteral();
  Token lexBinLiteral();
  Token lexOctLiteral();
  Token lexHexLiteral();
  Token lexDecOrFloatLiteral();
  Token lexDecimalLiteral();

  Token lexIdentifierOrKeyWord();
  Token lexIdentifierOrUnknownPrefix();
  Token lexFakeIdentifierOrUnknownPrefix();

  Location getLocation();

  void lineComment();
  void blockComment();

  bool isWhiteSpace(UChar32);
  void skipWhiteSpace();

  bool isASCII();
  bool isEmoji();

  bool isIdStart(int i = 0);
  bool isIdContinue(int i = 0);
  UChar32 getUchar(int i = 0);
  void skip();
  UChar32 peek(int i = 0);

  uint32_t lineNumber;
  uint32_t columnNumber;

  CheckPoint getCheckPoint() const;
  void recover(const CheckPoint&);
};

} // namespace rust_compiler::lexer


/*

  C API: New API for Unicode Normalization
 */
