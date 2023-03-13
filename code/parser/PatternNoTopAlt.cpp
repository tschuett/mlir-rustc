#include "AST/MacroInvocationSemiStatement.h"
#include "AST/OuterAttribute.h"
#include "AST/PathExpression.h"
#include "AST/Patterns/GroupedPattern.h"
#include "AST/Patterns/MacroInvocationPattern.h"
#include "AST/Patterns/PathPattern.h"
#include "AST/Patterns/ReferencePattern.h"
#include "AST/Patterns/StructPattern.h"
#include "AST/Patterns/StructPatternElements.h"
#include "AST/Patterns/TuplePattern.h"
#include "AST/Patterns/TuplePatternItems.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Patterns/TupleStructPattern.h"
#include "AST/Patterns/WildcardPattern.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::patterns;
using namespace llvm;

namespace rust_compiler::parser {

/// https://doc.rust-lang.org/reference/patterns.html

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseMacroInvocationPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  MacroInvocationPattern pattern = {loc};

  StringResult<ast::SimplePath> simplePath = parseSimplePath();
  if (!simplePath) {
    llvm::errs() << "failed to parse simple path in macro invocation pattern: "
                 << simplePath.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  pattern.setPath(simplePath.getValue());

  if (!check(TokenKind::Not)) {
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse macro invocation pattern");
  }
  assert(eat(TokenKind::Not));

  StringResult<std::shared_ptr<ast::DelimTokenTree>> token =
      parseDelimTokenTree();
  if (!token) {
    llvm::errs()
        << "failed to parse delim token tree in macro invocation pattern: "
        << token.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  pattern.setTree(token.getValue());

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<MacroInvocationPattern>(pattern));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseWildCardPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  WildcardPattern pat = {loc};

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<WildcardPattern>(pat));
  }

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      "failed to parse wild card pattern");
}

StringResult<TuplePatternItems> Parser::parseTuplePatternItems() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};

  llvm::errs() << "parseTuplePatternItems"
               << "\n";

  Location loc = getLocation();
  TuplePatternItems items = {loc};

  if (check(TokenKind::DotDot)) {
    items.setRestPattern();
    return StringResult<TuplePatternItems>(items);
  }

  StringResult<std::shared_ptr<ast::patterns::Pattern>> first = parsePattern();
  if (!first) {
    llvm::errs() << "failed to parse first pattern in tuple pattern items: "
                 << first.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse first pattern in tuple pattern items",
                      first.getError())
            .str();
    return StringResult<TuplePatternItems>(s);
    // exit(EXIT_FAILURE);
  }
  items.addPattern(first.getValue());

  if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
    items.setTrailingComma();
    assert(eat(TokenKind::Comma));
    return StringResult<TuplePatternItems>(items);
  } else if (check(TokenKind::ParenClose)) {
    return StringResult<TuplePatternItems>(items);
  }

  assert(eat(TokenKind::Comma));

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<TuplePatternItems>(
          "failed to parse tuple pattern items: eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return StringResult<TuplePatternItems>(items);
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<TuplePatternItems>(items);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
    } else {
      StringResult<std::shared_ptr<ast::patterns::Pattern>> next =
          parsePattern();

      if (!next) {
        llvm::errs() << "failed to parse next pattern in tuple pattern items: "
                     << next.getError() << "\n";
        printFunctionStack();
        std::string s =
            llvm::formatv("{0} {1}",
                          "failed to parse next pattern in tuple pattern items",
                          next.getError())
                .str();
        return StringResult<TuplePatternItems>(s);
        // exit(EXIT_FAILURE);
      }
      items.addPattern(next.getValue());
    }
  }
  return StringResult<TuplePatternItems>("failed to parse tuple pattern items");
}

Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
Parser::parseTupleOrTupleStructPattern() {
  llvm::errs() << "parseTupleOrTupleStructPattern"
               << "\n";

  if (check(TokenKind::ParenOpen))
    return parseTuplePattern();

  return parseTupleStructPattern();
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTuplePattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TuplePattern tuple = {loc};

  llvm::errs() << "parseTuplePattern"
               << "\n";

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ( token in tuple pattern");
  assert(eat(TokenKind::ParenOpen));

  StringResult<TuplePatternItems> items = parseTuplePatternItems();
  if (!items) {
    llvm::errs() << "failed to parse tuple pattern items in tuple pattern: "
                 << items.getError() << "\n";
    printFunctionStack();
    std::string s =
        llvm::formatv("{0} {1}",
                      "failed to parse tuple pattern items in tuple pattern: ",
                      items.getError())
            .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
    // exit(EXIT_FAILURE);
  }
  tuple.setItems(items.getValue());

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ) token in tuple pattern");
  assert(eat(TokenKind::ParenClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<TuplePattern>(tuple));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseGroupedPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  GroupedPattern grouped = {loc};

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ( token in grouped pattern");
  assert(eat(TokenKind::ParenOpen));

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in grouped pattern: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  grouped.setPattern(pattern.getValue());

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ) token in grouped pattern");
  assert(eat(TokenKind::ParenClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<GroupedPattern>(grouped));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseMacroInvocationOrPathOrStructOrTupleStructPattern() {

  while (true) {
  }
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseGroupedOrTuplePattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  CheckPoint cp = getCheckPoint();

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ( token in grouped or tuple pattern");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::DotDot)) {
    recover(cp);
    return parseTuplePattern();
  }

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse pattern in grouped or tuple pattern: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse pattern in grouped or tuple pattern");
    // exit(EXIT_FAILURE);
  }

  if (check(TokenKind::ParenClose)) {
    recover(cp);
    return parseGroupedPattern();
  }

  return parseTuplePattern();
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePatternWithoutRange() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  //  CheckPoint cp = getCheckPoint();

  if (checkLiteral()) {
    return parseLiteralPattern();
  } else if (check(TokenKind::Underscore)) {
    return parseWildCardPattern();
  } else if (check(TokenKind::DotDot)) {
    return parseRestPattern();
  } else if (check(TokenKind::And)) {
    return parseReferencePattern();
  } else if (check(TokenKind::AndAnd)) {
    return parseReferencePattern();
  } else if (check(TokenKind::SquareOpen)) {
    return parseSlicePattern();
  } else if (check(TokenKind::ParenOpen)) {
    return parseGroupedOrTuplePattern();
  }

  return parseMacroInvocationOrPathOrStructOrTupleStructPattern();
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePathOrStructOrTupleStructPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();

  //  // PathSep
  //  if (check(TokenKind::PathSep))
  //    assert(eat(TokenKind::PathSep));

  if (check(TokenKind::Lt))
    return parsePathPattern();

  StringResult<std::shared_ptr<ast::Expression>> pathIn =
      parsePathInExpression();
  if (!pathIn) {
    llvm::errs() << "failed to parse path in expression in "
                    "parsePathOrStructOrTupleStructPattern: "
                 << pathIn.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  if (check(TokenKind::ParenOpen)) {
    recover(cp);
    return parseTupleStructPattern();
  } else if (check(TokenKind::BraceOpen)) {
    recover(cp);
    return parseStructPattern();
  } else {
    PathPattern pat = {loc};
    pat.setPath(pathIn.getValue());
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<PathPattern>(pat));
  }
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePathPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  PathPattern path = {loc};

  llvm::outs() << "parsePathPattern"
               << "\n";

  StringResult<std::shared_ptr<ast::Expression>> pathExpr =
      parsePathExpression();
  if (!pathExpr) {
    llvm::errs() << "failed to parse path  expression in "
                    "parse path pattern: "
                 << pathExpr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  path.setPath(pathExpr.getValue());

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<PathPattern>(path));
}


StringResult<ast::patterns::TupleStructItems> Parser::parseTupleStructItems() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TupleStructItems items = {loc};

  llvm::errs() << "parseTupleStructItems"
               << "\n";

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
      parsePattern();
  if (!pattern) {
    llvm::errs() << "failed to parse  pattern in "
                    "parse tuple struct item1s: "
                 << pattern.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s = llvm::formatv("{0} {1}",
                                  "failed to parse  pattern in "
                                  "parse tuple struct item1s: ",
                                  pattern.getError())
                        .str();
    return StringResult<ast::patterns::TupleStructItems>(s);
  }
  items.addPattern(pattern.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::patterns::TupleStructItems>(
          "failed to parse tuple struct items field: eof");
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<ast::patterns::TupleStructItems>(items);
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose)) {
      items.setTrailingComma();
      assert(eat(TokenKind::Comma));
      return StringResult<ast::patterns::TupleStructItems>(items);
    } else if (check(TokenKind::Comma) && !check(TokenKind::ParenClose)) {
      assert(eat(TokenKind::Comma));
      StringResult<std::shared_ptr<ast::patterns::Pattern>> pattern =
          parsePattern();
      if (!pattern) {
        llvm::errs() << "failed to parse  pattern in "
                        "parse tuple struct ite2ms: "
                     << pattern.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      items.addPattern(pattern.getValue());
    } else {
      return StringResult<ast::patterns::TupleStructItems>(
          "failed to parse tuple struct items field");
    }
  }
  return StringResult<ast::patterns::TupleStructItems>(
      "failed to parse tuple struct items field");
}


StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseTupleStructPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  TupleStructPattern pat = {loc};

  llvm::errs() << "parseTupleStructPattern"
               << "\n";

  StringResult<std::shared_ptr<ast::Expression>> path = parsePathInExpression();
  if (!path) {
    llvm::errs() << "failed to parse path in expression in "
                    "parse tuple struct pattern: "
                 << path.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s = llvm::formatv("{0} {1}",
                                  "failed to parse path in expression in "
                                  "parse tuple struct pattern: ",
                                  path.getError())
                        .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
  }
  pat.setPath(path.getValue());

  if (!check(TokenKind::ParenOpen)) {
    llvm::errs() << Token2String(getToken().getKind()) << "\n";
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ( token in tuple struct pattern");
  }
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<TupleStructPattern>(pat));
  }

  StringResult<ast::patterns::TupleStructItems> items = parseTupleStructItems();
  if (!items) {
    llvm::errs() << "failed to parse tuple struct items in "
                    "parse tuple struct pattern: "
                 << items.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s = llvm::formatv("{0} {1}",
                                  "failed to parse tuple struct items in "
                                  "parse tuple struct pattern: ",
                                  items.getError())
                        .str();
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
  }
  pat.setItems(items.getValue());

  if (!check(TokenKind::ParenClose))
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse ) token tuple struct pattern");
  assert(eat(TokenKind::ParenClose));

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<TupleStructPattern>(pat));
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


StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseReferencePattern() {
  if (!check(lexer::TokenKind::And) && !check(lexer::TokenKind::AndAnd)) {
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        "failed to parse reference pattern");
  }

  Location loc = getLocation();

  ReferencePattern refer = {loc};

  if (check(lexer::TokenKind::And)) {
    assert(eat(lexer::TokenKind::And));
    refer.setAnd();
  } else if (check(lexer::TokenKind::AndAnd)) {
    assert(eat(lexer::TokenKind::AndAnd));
    refer.setAndAnd();
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(lexer::KeyWordKind::KW_MUT));
    refer.setMut();
  }

  StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>> woRange =
      parsePatternWithoutRange();
  if (!woRange) {
    llvm::errs() << "failed to parse pattern without range in "
                    "parse reference pattern: "
                 << woRange.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  refer.setPattern(woRange.getValue());

  return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
      std::make_shared<ReferencePattern>(refer));
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern() {
  Location loc = getLocation();

  llvm::errs()
      << "parseRangeOrIdentifierOrStructOrTupleStructOrMacroInvocationPattern"
      << "\n";

  CheckPoint point = getCheckPoint();

  if (check(TokenKind::Identifier) && check(TokenKind::At, 1)) {
    return parseIdentifierPattern();
  } else if (checkKeyWord(KeyWordKind::KW_REF)) {
    return parseIdentifierPattern();
  } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
    return parseIdentifierPattern();
  }

  /*
     else if (check(TokenKind::Identifier)) {
    return parseIdentifierOrPathPattern();
  }
  */

  /*
    PathExpression   -> RangePattern, PathPattern
    PathInExpression -> StructPattern, TupleStructPattern
    SimplePath       -> MacroInvocation
   */

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          "failed to parse "
          "RangeOrIdentifierOrStructOrTupleStructOrMacroIn"
          "vocationPattern: eof");
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
      return parseMacroInvocationPattern();
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
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::Comma)) {
      // GenericArgs
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::Eq)) {
      // terminator
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::Colon)) {
      // terminator
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::Semi)) {
      // terminator
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::INTEGER_LITERAL)) {
      // literal
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else if (check(TokenKind::STRING_LITERAL)) {
      // literal
      recover(point);
      return parseLiteralPattern();
    } else if (check(TokenKind::ParenClose)) {
      // terminator
      recover(point);
      return parsePathOrStructOrTupleStructPattern();
    } else {
      // error
      std::string s =
          llvm::formatv(
              "{0} {1}",
              "failed to parse RangeOrIdentifierOrStructOrTupleStructOrMa"
              "croInvocationPattern with unknown token",
              Token2String(getToken().getKind()))
              .str();
      llvm::outs() << Token2String(getToken().getKind()) << "\n";
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
    }
  }
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parsePatternNoTopAlt() {

  llvm::errs() << "parsePatternNoTopAlt: " << Token2String(getToken().getKind())
               << "\n";

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
  } else if (checkLiteral()) {
    return parseLiteralPattern();
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

adt::Result<std::shared_ptr<ast::patterns::PatternNoTopAlt>, std::string>
Parser::parseIdentifierOrPathPattern() {
  if (checkIdentifier() && (getToken(1).getKind() == TokenKind::PathSep))
    return parsePathPattern();
  return parseIdentifierPattern();
}

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
