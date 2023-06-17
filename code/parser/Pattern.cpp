#include "AST/Patterns/GroupedPattern.h"
#include "AST/Patterns/LiteralPattern.h"
#include "AST/Patterns/RangePattern.h"
#include "AST/Patterns/RestPattern.h"
#include "AST/Patterns/SlicePattern.h"
#include "AST/Patterns/TuplePattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRestPattern() {
  Location loc = getLocation();

  RestPattern pattern = {loc};

  if (check(TokenKind::DotDot)) {
    assert(eat(TokenKind::DotDot));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<RestPattern>(pattern));
  }

  llvm::errs() << "failed to parse rest pattern: unknown token "
               << Token2String(getToken().getKind()) << "\n";

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      "failed to parse rest pattern: unknown token");
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
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
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse identifier in identifier pattern");
  }

  Token tok = getToken();
  pattern.setIdentifier(tok.getIdentifier());
  assert(eat(TokenKind::Identifier));

  if (check(TokenKind::At)) {
    assert(eat(TokenKind::At));

    StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
        patternNoTopAlt = parsePatternNoTopAlt();
    if (!patternNoTopAlt) {
      llvm::errs()
          << "failed to parse pattern to top alt in identifier pattern: "
          << patternNoTopAlt.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    pattern.addPattern(patternNoTopAlt.getValue());
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<IdentifierPattern>(pattern));
  }

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<IdentifierPattern>(pattern));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseLiteralPattern() {
  Location loc = getLocation();

  LiteralPattern pattern = {loc};

  if (checkKeyWord(KeyWordKind::KW_TRUE)) {
    pattern.setKind(LiteralPatternKind::True, getToken().getStorage());
    assert(eatKeyWord(KeyWordKind::KW_TRUE));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (checkKeyWord(KeyWordKind::KW_FALSE)) {
    pattern.setKind(LiteralPatternKind::False, getToken().getStorage());
    assert(eatKeyWord(KeyWordKind::KW_FALSE));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::CHAR_LITERAL)) {
    pattern.setKind(LiteralPatternKind::CharLiteral, getToken().getStorage());
    assert(eat(TokenKind::CHAR_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::BYTE_LITERAL)) {
    pattern.setKind(LiteralPatternKind::ByteLiteral, getToken().getStorage());
    assert(eat(TokenKind::BYTE_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::StringLiteral, getToken().getStorage());
    assert(eat(TokenKind::STRING_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::RAW_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::RawStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::RAW_STRING_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::BYTE_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::ByteStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::BYTE_STRING_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::RAW_BYTE_STRING_LITERAL)) {
    pattern.setKind(LiteralPatternKind::RawByteStringLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::RAW_BYTE_STRING_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::INTEGER_LITERAL)) {
    pattern.setKind(LiteralPatternKind::IntegerLiteral,
                    getToken().getStorage());
    assert(eat(TokenKind::INTEGER_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::FLOAT_LITERAL)) {
    pattern.setKind(LiteralPatternKind::FloatLiteral, getToken().getStorage());
    assert(eat(TokenKind::FLOAT_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::Minus) && check(TokenKind::INTEGER_LITERAL, 1)) {
    pattern.setKind(LiteralPatternKind::IntegerLiteral,
                    getToken().getStorage());
    pattern.setLeadingMinus();
    assert(eat(TokenKind::Minus));
    assert(eat(TokenKind::INTEGER_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  } else if (check(TokenKind::Minus) && check(TokenKind::FLOAT_LITERAL, 1)) {
    pattern.setKind(LiteralPatternKind::FloatLiteral, getToken().getStorage());
    pattern.setLeadingMinus();
    assert(eat(TokenKind::Minus));
    assert(eat(TokenKind::FLOAT_LITERAL));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<LiteralPattern>(pattern));
  }
  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      "failed to parse literal pattern");
}

StringResult<std::shared_ptr<ast::patterns::Pattern>> Parser::parsePattern() {
  Location loc = getLocation();

  Pattern pattern = {loc};

  if (check(TokenKind::Or)) {
    assert(eat(TokenKind::Or));
    pattern.setLeadingOr();
  }

  StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>> first =
      parsePatternNoTopAlt();
  if (!first) {
    llvm::outs() << "failed to parse pattern to top alt in pattern: "
                 << first.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse not top alt pattern in parse pattern: first",
            first.getError())
            .str();
    return StringResult<std::shared_ptr<ast::patterns::Pattern>>(s);
  }
  pattern.addPattern(first.getValue());

  if (check(TokenKind::Or)) {
    assert(eat(TokenKind::Or));

    while (true) {
      StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
          patternNoTopAlt = parsePatternNoTopAlt();
      if (!patternNoTopAlt) {
        llvm::outs() << "failed to parse pattern to top alt in pattern: "
                     << patternNoTopAlt.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv(
                "{0} {1}",
                "failed to parse not top alt pattern in parse pattern: next",
                patternNoTopAlt.getError())
                .str();
        return StringResult<std::shared_ptr<ast::patterns::Pattern>>(s);
        // exit(EXIT_FAILURE);
      }
      pattern.addPattern(patternNoTopAlt.getValue());
      if (check(TokenKind::Or)) {
        assert(eat(TokenKind::Or));
        continue;
      } else if (check(TokenKind::Eof)) {
        return StringResult<std::shared_ptr<ast::patterns::Pattern>>(
            "found eof in  pattern");

      } else {
        return StringResult<std::shared_ptr<ast::patterns::Pattern>>(
            std::make_shared<Pattern>(pattern));
      }
    }
  }
  return StringResult<std::shared_ptr<ast::patterns::Pattern>>(
      std::make_shared<Pattern>(pattern));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTupleOrGroupedPattern() {
  Location loc = getLocation();

//  llvm::errs() << "parseTupleOrGroupedPattern"
//               << "\n";

  if (!check(TokenKind::ParenOpen)) {
    llvm::errs() << "failed to parse token ( in tuple or group pattern "
                 << "\n";
    // report error
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse (");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::DotDot) && check(TokenKind::ParenClose, 1)) {
    TuplePattern tuple = {loc};

    assert(eat(TokenKind::DotDot));
    assert(eat(TokenKind::ParenClose));

    TuplePatternItems items = {loc};
    items.setRestPattern();
    tuple.setItems(items);

    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<TuplePattern>(tuple));
  }

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in tuple or group: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse pattern in tuple or group pattern: first pattern");
    // exit(EXIT_FAILURE);
  }
  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    // done GroupedPattern
    GroupedPattern group = {loc};
    group.setPattern(pattern.getValue());
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<GroupedPattern>(group));
  } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    assert(eat(TokenKind::Comma));
    assert(eat(TokenKind::ParenClose));
    TuplePatternItems items = {loc};

    items.addPattern(pattern.getValue());
    items.setTrailingComma();

    TuplePattern tuple = {loc};
    tuple.setItems(items);

    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<TuplePattern>(tuple));
  } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose, 1)) {
    // continue
    assert(eat(TokenKind::Comma));
  } else {
    // report
    // error ?
    llvm::outs() << "found unexpected token in tuple or grouped pattern"
                 << Token2String(getToken().getKind()) << "\n";
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse pattern in tuple or group pattern: unexpected token");
  }

  TuplePatternItems items = {loc};

  items.addPattern(pattern.getValue());

  while (true) {
    StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
        parsePattern();
    if (!pattern) {
      llvm::errs() << "failed to parse pattern in turple or grouped pattern: "
                   << pattern.getError() << "\n";
      printFunctionStack();

      std::string S = llvm::formatv("{0} {1}",
                                    "failed to parse pattern in tuple or group "
                                    "pattern, in while loop: ",
                                    pattern.getError())
                          .str();
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(S);
      // exit(EXIT_FAILURE);
    }
    items.addPattern(pattern.getValue());

    if (check(TokenKind::ParenClose)) {
      assert(eat(TokenKind::ParenClose));
      // done
      TuplePattern pattern = {loc};
      pattern.setItems(items);
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          std::make_shared<TuplePattern>(pattern));
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      assert(eat(TokenKind::ParenClose));
      TuplePattern pattern = {loc};
      pattern.setItems(items);
      items.setTrailingComma();
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          std::make_shared<TuplePattern>(pattern));
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      continue;
    } else if (check(TokenKind::Eof)) {
      // abort
    }
  }
}

} // namespace rust_compiler::parser
