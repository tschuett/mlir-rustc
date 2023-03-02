#include "AST/Patterns/GroupedPattern.h"
#include "AST/Patterns/LiteralPattern.h"
#include "AST/Patterns/RangePattern.h"
#include "AST/Patterns/RestPattern.h"
#include "AST/Patterns/SlicePattern.h"
#include "AST/Patterns/TuplePattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRestPattern() {
  Location loc = getLocation();

  RestPattern pattern = {loc};

  return std::make_shared<RestPattern>(pattern);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseIdentifierPattern() {
  Location loc = getLocation();

  IdentifierPattern pattern = {loc};

  if (checkKeyWord(KeyWordKind::KW_REF)) {
    assert(eatKeyWord(KeyWordKind::KW_REF));
    pattern.setRef();
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(KeyWordKind::KW_MUT));
    pattern.setMut();
  }

  if (!check(TokenKind::Identifier)) {
    // report errro
  }

  Token tok = getToken();
  pattern.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::At)) {
    assert(eat(TokenKind::At));

    llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
        patternNoTopAlt = parsePatternNoTopAlt();
    if (auto e = patternNoTopAlt.takeError()) {
      llvm::errs()
          << "failed to parse pattern no top alt in identifier pattern : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    pattern.addPattern(*patternNoTopAlt);
    return std::make_shared<IdentifierPattern>(pattern);
  }

  return std::make_shared<IdentifierPattern>(pattern);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseLiteralPattern() {
  Location loc = getLocation();

  LiteralPattern pattern = {loc};

  if (checkKeyWord(KeyWordKind::KW_TRUE)) {
    pattern.setKind(LiteralPatternKind::True, getToken().getStorage());
    assert(eatKeyWord(KeyWordKind::KW_TRUE));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (checkKeyWord(KeyWordKind::KW_FALSE)) {
    pattern.setKind(LiteralPatternKind::False, getToken().getStorage());
    assert(eatKeyWord(KeyWordKind::KW_FALSE));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::CHAR_LITERAL)) {
    pattern.setKind(LiteralPatternKind::CharLiteral, getToken().getStorage());
    assert(eat(TokenKind::CHAR_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::BYTE_LITERAL)) {
    pattern.setKind(LiteralPatternKind::ByteLiteral, getToken().getStorage());
    assert(eat(TokenKind::BYTE_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::StringLiteral, getToken().getStorage());
    assert(eat(TokenKind::STRING_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::RAW_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::RawStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::RAW_STRING_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::BYTE_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::ByteStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::BYTE_STRING_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::RAW_BYTE_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::RawByteStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::RAW_BYTE_STRING_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::INTEGER_LITERAL)) {
    pattern.setKind(LiteralPatternKind::IntegerLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::INTEGER_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::FLOAT_LITERAL)) {
    pattern.setKind(LiteralPatternKind::FloatLiteral, getToken().getStorage());
    assert(eat(TokenKind::FLOAT_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::Minus) && check(TokenKind::INTEGER_LITERAL, 1)) {
    pattern.setKind(LiteralPatternKind::IntegerLiteral,
                    getToken().getStorage());
    pattern.setLeadingMinus();
    assert(eat(TokenKind::Minus));
    assert(eat(TokenKind::INTEGER_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  } else if (check(TokenKind::Minus) && check(TokenKind::FLOAT_LITERAL, 1)) {
    pattern.setKind(LiteralPatternKind::FloatLiteral, getToken().getStorage());
    pattern.setLeadingMinus();
    assert(eat(TokenKind::Minus));
    assert(eat(TokenKind::FLOAT_LITERAL));
    return std::make_shared<LiteralPattern>(pattern);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse literal pattern");
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRangePattern() {
  Location loc = getLocation();

  RangePattern pattern = {loc};

  assert(false);
}

llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> Parser::parsePattern() {
  Location loc = getLocation();

  Pattern pattern = {loc};

  if (check(TokenKind::Or)) {
    assert(check(TokenKind::Or));
    pattern.setLeadingOr();
  }

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> first =
      parsePatternNoTopAlt();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt in pattern : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  pattern.addPattern(*first);

  if (check(TokenKind::Or)) {
    assert(check(TokenKind::Or));

    while (true) {
      llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
          patternNoTopAlt = parsePatternNoTopAlt();
      if (auto e = patternNoTopAlt.takeError()) {
        llvm::errs() << "failed to parse pattern no top alt in pattern : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      pattern.addPattern(*patternNoTopAlt);
      if (check(TokenKind::Or)) {
        assert(check(TokenKind::Or));
        continue;
      } else if (check(TokenKind::Eof)) {
        return createStringError(inconvertibleErrorCode(),
                                 "found eof in  pattern");

      } else {
        return std::make_shared<Pattern>(pattern);
      }
    }
  }
  return std::make_shared<Pattern>(pattern);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTupleOrGroupedPattern() {
  Location loc = getLocation();

  if (!check(TokenKind::ParenOpen)) {
    // report error
  }
  assert(check(TokenKind::ParenOpen));

  if (check(TokenKind::DotDot) && check(TokenKind::ParenClose, 1)) {
    TuplePattern tuple = {loc};

    assert(check(TokenKind::DotDot));
    assert(check(TokenKind::ParenClose));

    TuplePatternItems items = {loc};
    items.setRestPattern();
    tuple.setItems(items);

    return std::make_shared<TuplePattern>(tuple);
  }

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (auto e = pattern.takeError()) {
    llvm::errs()
        << "failed to parse pattern no top alt in tuple or grouped patten  : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::ParenClose)) {
    assert(check(TokenKind::ParenClose));
    // done GroupedPattern
    GroupedPattern group = {loc};
    group.setPattern(*pattern);
    return std::make_shared<GroupedPattern>(group);
  } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    TuplePatternItems items = {loc};

    items.addPattern(*pattern);
    items.setTrailingComma();

    TuplePattern tuple = {loc};
    tuple.setItems(items);

    return std::make_shared<TuplePattern>(tuple);
  } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
    // continue
  } else {
    // report error ?
  }

  TuplePatternItems items = {loc};

  items.addPattern(*pattern);

  while (true) {
    llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
        parsePattern();
    if (auto e = pattern.takeError()) {
      llvm::errs() << "failed to parse pattern no top alt in tuple or "
                      "grouped patten  : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    items.addPattern(*pattern);

    if (check(TokenKind::ParenClose)) {
      // done
      TuplePattern pattern = {loc};
      pattern.setItems(items);
      return std::make_shared<TuplePattern>(pattern);
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(check(TokenKind::Comma));
      TuplePattern pattern = {loc};
      pattern.setItems(items);
      items.setTrailingComma();
      return std::make_shared<TuplePattern>(pattern);
    } else if (check(TokenKind::Comma)) {
      assert(check(TokenKind::Comma));
      continue;
    } else if (check(TokenKind::Eof)) {
      // abort
    }
  }
}

llvm::Expected<ast::patterns::SlicePatternItems>
Parser::parseSlicePatternItems() {
  Location loc = getLocation();

  SlicePatternItems items = {loc};

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> first =
      parsePattern();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse pattern items in slice pattern items : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  items.addPattern(*first);

  // TODO

  assert(false);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseSlicePattern() {
  Location loc = getLocation();

  SlicePattern slice = {loc};

  if (!check(TokenKind::SquareOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse [ token in slice pattern");
  }
  assert(check(TokenKind::SquareOpen));

  if (check(TokenKind::SquareClose)) {
    // done
    assert(check(TokenKind::SquareClose));
    return std::make_shared<SlicePattern>(slice);
  }

  llvm::Expected<ast::patterns::SlicePatternItems> items =
      parseSlicePatternItems();
  if (auto e = items.takeError()) {
    llvm::errs() << "failed to parse slice pattern items in slice pattern : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  slice.setPatternItems(*items);

  assert(check(TokenKind::SquareClose));

  return std::make_shared<SlicePattern>(slice);
}

} // namespace rust_compiler::parser
