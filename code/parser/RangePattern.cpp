#include "AST/Patterns/RangePattern.h"

#include "AST/Patterns/RangePatternBound.h"
#include "AST/TokenTree.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::patterns;

namespace rust_compiler::parser {

StringResult<RangePatternBound> Parser::parseRangePatternBound() {
  RangePatternBound bound = {getLocation()};

  if (check(TokenKind::CHAR_LITERAL)) {
    bound.setKind(RangePatternBoundKind::CharLiteral);
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
  } else if (check(TokenKind::BYTE_LITERAL)) {
    bound.setKind(RangePatternBoundKind::ByteLiteral);
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
  } else if (check(TokenKind::INTEGER_LITERAL)) {
    bound.setKind(RangePatternBoundKind::IntegerLiteral);
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
  } else if (check(TokenKind::Minus) && check(TokenKind::INTEGER_LITERAL, 1)) {
    bound.setKind(RangePatternBoundKind::MinusIntegerLiteral);
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
    assert(eat(getToken().getKind()));
  } else if (check(TokenKind::FLOAT_LITERAL)) {
    bound.setKind(RangePatternBoundKind::FloatLiteral);
    assert(eat(getToken().getKind()));
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
  } else if (check(TokenKind::Minus) && check(TokenKind::FLOAT_LITERAL, 1)) {
    bound.setKind(RangePatternBoundKind::MinusFloatLitera);
    bound.setStorage(getToken().getStorage());
    assert(eat(getToken().getKind()));
    assert(eat(getToken().getKind()));
  } else {
    // PathExpression
    Result<std::shared_ptr<ast::Expression>, std::string> path =
        parsePathExpression();
    if (!path) {
      llvm::errs() << "failed to parse path expression in range pattern bound"
                   << path.getError() << "\n";
      std::string s =
          llvm::formatv(
              "{0}\n{1}",
              "failed to parse path expression in range pattern bound",
              path.getError())
              .str();
      return StringResult<RangePatternBound>(s);
    }
    bound.setKind(RangePatternBoundKind::PathExpression);
    bound.setPath(path.getValue());
  }
  return StringResult<RangePatternBound>(bound);
}

StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>
Parser::parseRangePattern() {
  Location loc = getLocation();

  RangePattern pattern = {loc};

  if (check(TokenKind::DotDotEq)) {
    assert(eat(TokenKind::DotDotEq));
    StringResult<RangePatternBound> upper = parseRangePatternBound();
    if (!upper) {
      // report error
      llvm::errs()
          << "failed to parse upper range pattern bound in range pattern: "
          << upper.getError() << "\n";
      std::string s =
          llvm::formatv(
              "{0}\n{1}",
              "failed to parse upper range pattern bound in range pattern: ",
              upper.getError())
              .str();
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
    }
    pattern.setUpper(upper.getValue());
    pattern.setKind(RangePatternKind::HalfOpenRangePattern);
    return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
        std::make_shared<RangePattern>(pattern));
  } else { // start with lower bound
    StringResult<RangePatternBound> lower = parseRangePatternBound();
    if (!lower) {
      // report error
      llvm::errs()
          << "failed to parse lower range pattern bound in range pattern: "
          << lower.getError() << "\n";
      std::string s =
          llvm::formatv(
              "{0}\n{1}",
              "failed to parse lower range pattern bound in range pattern: ",
              lower.getError())
              .str();
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
    }
    pattern.setLower(lower.getValue());

    if (check(TokenKind::DotDotEq)) {
      assert(eat(TokenKind::DotDotEq));
      StringResult<RangePatternBound> upper = parseRangePatternBound();
      if (!upper) {
        // report error
        llvm::errs()
            << "failed to parse upper range pattern bound in range pattern: "
            << upper.getError() << "\n";
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse upper range pattern bound in range pattern: ",
                upper.getError())
                .str();
        return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
      }
      pattern.setKind(RangePatternKind::InclusiveRangePattern);
      pattern.setUpper(upper.getValue());
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          std::make_shared<RangePattern>(pattern));
    } else if (check(TokenKind::DotDot)) {
      assert(eat(TokenKind::DotDot));
      // done
      pattern.setKind(RangePatternKind::HalfOpenRangePattern);
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          std::make_shared<RangePattern>(pattern));
    } else if (check(TokenKind::DotDotDot)) {
      assert(eat(TokenKind::DotDotDot));
      StringResult<RangePatternBound> upper = parseRangePatternBound();
      if (!upper) {
        // report error
        llvm::errs()
            << "failed to parse upper range pattern bound in range pattern: "
            << upper.getError() << "\n";
        std::string s =
            llvm::formatv(
                "{0}\n{1}",
                "failed to parse upper range pattern bound in range pattern: ",
                upper.getError())
                .str();
        return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(s);
        // report error
      }
      pattern.setUpper(upper.getValue());
      pattern.setKind(RangePatternKind::ObsoleteRangePattern);
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          std::make_shared<RangePattern>(pattern));
    } else {
      // report error
      llvm::errs() << "failed to parse range pattern: "
                   << Token2String(getToken().getKind()) << "\n";
      return StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>>(
          "failed to parse range pattern");
    }
  }

  assert(false);
}

} // namespace rust_compiler::parser
