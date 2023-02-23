#include "AST/Patterns/ReferencePattern.h"
#include "AST/Patterns/StructPattern.h"
#include "AST/Patterns/StructPatternElements.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::patterns;
using namespace llvm;

namespace rust_compiler::parser {

/// https://doc.rust-lang.org/reference/patterns.html

 llvm::Expected<ast::patterns::StructPatternElements>
 Parser::parseStructPatternElements() {
   Location loc = getLocation();
   StructPatternElements elements = {loc};

   CheckPoint cp = getCheckPoint();

   if (checkOuterAttribute()) {
     llvm::Expected<std::vector<ast::OuterAttribute>> outer =
         parseOuterAttributes();
     if (auto e = outer.takeError()) {
       llvm::errs()
           << "failed to parse outer attributes in struct pattern elements : "
           << toString(std::move(e)) << "\n";
       exit(EXIT_FAILURE);
     }
   }

   if (check(TokenKind::DotDot)) {
     // StructPatternEtCetera
   } else {
   }
   xxx;
 }

llvm::Expected<ast::patterns::TupleStructItems>
Parser::parseTupleStructItems() {
  Location loc = getLocation();
  TupleStructItems items = {loc};

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern in tuple struct item : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  items.addPattern(*pattern);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse tuple struct items field: eof");
    } else if (check(TokenKind::ParenClose)) {
      return items;
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose)) {
      items.setTrailingComma();
      assert(eat(TokenKind::Comma));
      return items;
    } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose)) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pattern =
          parsePattern();
      if (auto e = pattern.takeError()) {
        llvm::errs() << "failed to parse pattern in tuple struct item : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      items.addPattern(*pattern);
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse tuple struct items field");
    }
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse tuple struct items field");
}

llvm::Expected<ast::patterns::StructPatternField>
Parser::parseStructPatternField() {
  Location loc = getLocation();

  StructPatternField field = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs()
          << "failed to parse outer attributes in struct pattern field : "
          << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    field.setOuterAttributes(*outer);
  }

  if (check(TokenKind::INTEGER_LITERAL) && check(TokenKind::Colon)) {
    field.setTupleIndex(getToken().getLiteral());
    assert(eat(TokenKind::INTEGER_LITERAL));
    assert(eat(TokenKind::Colon));
    llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> patterns =
        parsePattern();
    if (auto e = patterns.takeError()) {
      llvm::errs() << "failed to parse pattern in struct pattern field : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    field.setPattern(*patterns);
    field.setKind(StructPatternFieldKind::TupleIndex);
    return field;
  } else if (checkIdentifier() && check(TokenKind::Colon)) {
    field.setIdentifier(getToken().getIdentifier());
    assert(eat(TokenKind::Identifier));
    assert(eat(TokenKind::Colon));
    llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> patterns =
        parsePattern();
    if (auto e = patterns.takeError()) {
      llvm::errs() << "failed to parse pattern in struct pattern field : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    field.setPattern(*patterns);
    field.setKind(StructPatternFieldKind::Identifier);
    return field;
  } else if (checkKeyWord(KeyWordKind::KW_REF) ||
             checkKeyWord(KeyWordKind::KW_MUT) || checkIdentifier()) {
    if (checkKeyWord(KeyWordKind::KW_REF)) {
      field.setRef();
      assert(eatKeyWord(KeyWordKind::KW_REF));
    }
    if (checkKeyWord(KeyWordKind::KW_MUT)) {
      field.setMut();
      assert(eatKeyWord(KeyWordKind::KW_MUT));
    }
    if (!checkIdentifier())
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse identifier token in struct pattern field");
    field.setIdentifier(getToken().getIdentifier());
    field.setKind(StructPatternFieldKind::RefMut);
    assert(eat(TokenKind::Identifier));
    return field;
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse struct pattern field");
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse struct pattern field");
}

llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTupleStructPattern() {
  Location loc = getLocation();
  TupleStructPattern pat = {loc};

  llvm::Expected<std::shared_ptr<ast::PathExpression>> path =
      parsePathInExpression();
  if (auto e = path.takeError()) {
    llvm::errs()
        << "failed to parse path in expression in tuple struct pattern : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  pat.setPath(*path);

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse tuple struct pattern");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<TupleStructPattern>(pat);
  }

  llvm::Expected<ast::patterns::TupleStructItems> items =
      parseTupleStructItems();
  if (auto e = items.takeError()) {
    llvm::errs()
        << "failed to parse tuple struct items in tuple struct pattern : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  pat.setItems(*items);

  if (!check(TokenKind::ParenClose))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse tuple struct pattern");
  assert(eat(TokenKind::ParenClose));

  return std::make_shared<TupleStructPattern>(pat);
}

llvm::Expected<ast::patterns::StructPatternFields>
Parser::parseStructPatternFields() {
  Location loc = getLocation();

  StructPatternFields fields = {loc};

  llvm::Expected<ast::patterns::StructPatternField> first =
      parseStructPatternField();
  if (auto e = first.takeError()) {
    llvm::errs()
        << "failed to parse struct pattern field in struct pattern fields : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  fields.addPattern(*first);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse struct pattern fields: eof");
    } else if (check(TokenKind::Comma)) {
      assert(eat((TokenKind::Comma)));
      llvm::Expected<ast::patterns::StructPatternField> next =
          parseStructPatternField();
      if (auto e = next.takeError()) {
        llvm::errs() << "failed to parse struct pattern field in struct "
                        "pattern fields : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      fields.addPattern(*next);
    } else {
      // done
      return fields;
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse struct pattern fields");
}

// llvm::Expected<ast::patterns::StructPatternElements>
// Parser::parseStructPatternElements() {
//   Location loc = getLocation();
//   StructPatternElements el = {loc};
//
//   CheckPoint cp = getCheckPoint();
//
//   if (checkOuterAttribute()) {
//     llvm::Expected<std::vector<ast::OuterAttribute>> outer =
//         parseOuterAttributes();
//     if (auto e = outer.takeError()) {
//       llvm::errs() << "failed to parse outer attribute in struct "
//                       "pattern elements : "
//                    << toString(std::move(e)) << "\n";
//       exit(EXIT_FAILURE);
//     }
//     if (check(TokenKind::DotDot)) {
//     } else if (check(TokenKind::INTEGER_LITERAL) &&
//                check(TokenKind::Colon, 1)) {
//     } else if (checkIdentifier() && check(TokenKind::Colon, 1)) {
//     } else if (checkIdentifier()) {
//     } else if (checkKeyWord(KeyWordKind::KW_REF)) {
//     } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
//     }
//   } else if (check(TokenKind::DotDot)) {
//   } else if (check(TokenKind::INTEGER_LITERAL)) {
//   } else if (checkIdentifier() && check(TokenKind::Colon, 1)) {
//   } else if (checkIdentifier()) {
//   } else if (checkKeyWord(KeyWordKind::KW_REF)) {
//   } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
//   }
//
//   xxx;
// }

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
