#include "AST/Patterns/ReferencePattern.h"
#include "AST/Patterns/StructPattern.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace llvm;

namespace rust_compiler::parser {

/// https://doc.rust-lang.org/reference/patterns.html

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseStructPattern() {
  Location loc = getLocation();

  StructPattern pat = {loc};

  llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
      parsePathInExpression();
  if (auto e = path.takeError()) {
    llvm::errs() << "failed to parse path in expression in struct pattern : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  pat.setPath(*path);

  if (!check(lexer::TokenKind::BraceOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct pattern");
  }
  assert(eat(TokenKind::BraceOpen));

  if (check(TokenKind::BraceClose)) {
    assert(eat(TokenKind::BraceClose));
    return std::make_shared<StructPattern>(pat);
  }

  llvm::Expected<StructPatternElements> pattern = parseStructPatternElements();
  if (auto e = pattern.takeError()) {
    llvm::errs()
        << "failed to parse struct pattern elements in struct pattern : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  pat.setElements(*pattern);

  if (!check(lexer::TokenKind::BraceClose)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct pattern");
  }
  assert(eat(TokenKind::BraceClose));

  return std::make_shared<StructPattern>(pat);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseReferencePattern() {
  if (!check(lexer::TokenKind::And) && !check(lexer::TokenKind::AndAnd)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse reference pattern");
  }

  Location loc = getLocation();

  ReferencePattern refer = {loc};

  if (check(lexer::TokenKind::And)) {
    assert(eat(lexer::TokenKind::And));
    refer.setAnd();
  }

  if (check(lexer::TokenKind::AndAnd)) {
    assert(eat(lexer::TokenKind::AndAnd));
    refer.setAndAnd();
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(lexer::KeyWordKind::KW_MUT));
    refer.setMut();
  }

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> woRange =
      parsePatternWithoutRange();
  if (auto e = woRange.takeError()) {
    llvm::errs()
        << "failed to parse pattern without block in reference pattern : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  refer.setPattern(*woRange);

  return std::make_shared<ReferencePattern>(refer);
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern() {
  Location loc = getLocation();

  CheckPoint point = getCheckPoint();

  if (check(TokenKind::Identifier) && check(TokenKind::At, 1)) {
    return parseIdentifierPattern();
  } else if (check(TokenKind::Identifier) && !check(TokenKind::At, 1)) {
    return parsePathPattern(); // Path patterns take precedence over identifier
                               // patterns.
  }

  /*
    PathExpression   -> RangePattern, PathPattern
    PathInExpression -> StructPattern, TupleStructPattern
    SimplePath       -> MacroInvocation
   */

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
    } else if (checkPathIdentSegment()) {
      assert(eatPathIdentSegment());
    } else if (checkSimplePathSegment()) {
      assert(eatSimplePathSegment());
    } else if (check(TokenKind::BraceOpen)) {
      recover(point);
      return parseStructPattern();
    } else if (check(TokenKind::ParenOpen)) {
      recover(point);
      return parseTupleStructPattern();
    } else if (check(TokenKind::Not)) {
      recover(point);
      return parseMacroInvocation();
    } else if (check(TokenKind::DotDot)) {
      recover(point);
      return parseRangePattern();
    } else if (check(TokenKind::DotDotDot)) {
      recover(point);
      return parseRangePattern();
    } else if (check(TokenKind::DotDotEq)) {
      recover(point);
      return parseRangePattern();
    } else if (check(TokenKind::Lt)) {
      // GenericArgs
      recover(point);
      return parsePathOrStructOrTuplePattern();
    } else {
      // error
    }
  }
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePatternNoTopAlt() {

  if (check(TokenKind::And) || check(TokenKind::AndAnd)) {
    return parseReferencePattern();
  } else if (check(TokenKind::ParenOpen)) {
    return parseTupleOrGroupedPattern();
  } else if (check(TokenKind::SquareOpen)) {
    return parseSlicePattern();
  } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
    return parseIdentifierPattern();
  } else if (checkKeyWord(KeyWordKind::KW_REF)) {
    return parseIdentifierPattern();
  } else if (check(TokenKind::Underscore) &&
             check(TokenKind::INTEGER_LITERAL, 1)) {
    return parseLiteralPattern();
  } else if (check(TokenKind::Underscore) &&
             check(TokenKind::FLOAT_LITERAL, 1)) {
    return parseLiteralPattern();
  } else if (checkKeyWord(KeyWordKind::KW_TRUE)) {
    return parseLiteralPattern();
  } else if (checkKeyWord(KeyWordKind::KW_FALSE)) {
    return parseLiteralPattern();
  } else if (check(TokenKind::Underscore)) {
    return parseLiteralPattern();
  } else if (check(lexer::TokenKind::DotDot)) {
    return parseRestPattern();
  } else if (check(TokenKind::CHAR_LITERAL) && check(TokenKind::DotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::CHAR_LITERAL) && check(TokenKind::DotDotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::CHAR_LITERAL) && check(TokenKind::DotDotEq, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::BYTE_LITERAL) && check(TokenKind::DotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::BYTE_LITERAL) && check(TokenKind::DotDotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::BYTE_LITERAL) && check(TokenKind::DotDotEq, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::INTEGER_LITERAL) && check(TokenKind::DotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::INTEGER_LITERAL) &&
             check(TokenKind::DotDotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::INTEGER_LITERAL) &&
             check(TokenKind::DotDotEq, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::FLOAT_LITERAL) && check(TokenKind::DotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::FLOAT_LITERAL) &&
             check(TokenKind::DotDotDot, 1)) {
    return parseRangePattern();
  } else if (check(TokenKind::FLOAT_LITERAL) && check(TokenKind::DotDotEq, 1)) {
    return parseRangePattern();
  }

  return parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern();
  // PathExpression
  // PathInExpression
  // Identifier
  // SimplePath

  /*
    MacroInvocation
    PathPattern
    RangePattern
    IdentifierPattern
    StructPattern
    TupleStructPattern
   */
}
// if (check(lexer::TokenKind::DotDot))
//   return parseRestPattern();

//  if (check(lexer::TokenKind::Underscore))
//    return parseWildCardPattern(); // maybe range literal

} // namespace rust_compiler::parser

/*
  RangePattern
  StructPattern
  TupleStructPattern
  PathPattern
  MacroInvocation
  IdentifierPattern
 */

/*
    } else if (check(TokenKind::CHAR_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::BYTE_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::STRING_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::RAW_STRING_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::BYTE_STRING_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::RAW_BYTE_STRING_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::INTEGER_LITERAL)) {
  return parseLiteralPattern();
} else if (check(TokenKind::FLOAT_LITERAL)) {
  return parseLiteralPattern();
*/
