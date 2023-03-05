#include "AST/InfiniteLoopExpression.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LabelBlockExpression.h"
#include "AST/OuterAttribute.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace rust_compiler::adt;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkLoopLabel() {
  return (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon, 1));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseLabelBlockExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  LabelBlockExpression bloc = {loc};

  StringResult<std::shared_ptr<ast::Expression>> block = parseBlockExpression();
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in label block expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  bloc.setBlock(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<LabelBlockExpression>(bloc));
}

// Result<LoopLabel, std::string> Parser::parseLoopLabel() {}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseLoopExpression(std::span<OuterAttribute> outer) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  if (checkLoopLabel()) {
    if (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon, 1)) {
      std::string label = getToken().getStorage();
      assert(eat(TokenKind::LIFETIME_OR_LABEL));
      assert(eat(TokenKind::Colon));

      // FIXME

      if (checkKeyWord(KeyWordKind::KW_LOOP)) {
        return parseInfiniteLoopExpression(outer);
      } else if (checkKeyWord(KeyWordKind::KW_WHILE) &&
                 checkKeyWord(KeyWordKind::KW_LET, 1)) {
        return parsePredicatePatternLoopExpression(outer);
      } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
        return parsePredicateLoopExpression(outer);
      } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
        return parseIteratorLoopExpression(outer);
      } else {
        return parseLabelBlockExpression(outer);
      }
      return Result<std::shared_ptr<ast::Expression>, std::string>(
          "failed to parse loop expression with loop label");
    }
  }

  if (checkKeyWord(KeyWordKind::KW_LOOP)) {
    return parseInfiniteLoopExpression({});
  } else if (checkKeyWord(KeyWordKind::KW_WHILE) &&
             checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parsePredicatePatternLoopExpression({});
  } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
    return parsePredicateLoopExpression(outer);
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    return parseIteratorLoopExpression(outer);
  } else {
    return parseLabelBlockExpression(outer);
  }
  return Result<std::shared_ptr<ast::Expression>, std::string>(
      "failed to parse loop expression without loop label");
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseIteratorLoopExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  IteratorLoopExpression it = {loc};

  if (!checkKeyWord(KeyWordKind::KW_FOR))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse for keyword");
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pred = parsePattern();
  if (!pred) {
    llvm::errs() << "failed to parse pattern in iterator loop expression: "
                 << pred.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  it.setPattern(pred.getValue());

  if (!checkKeyWord(KeyWordKind::KW_IN))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse in keyword");
  assert(eatKeyWord(KeyWordKind::KW_IN));

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in iterator loop expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  it.setExpression(expr.getValue());

  StringResult<std::shared_ptr<ast::Expression>> block = parseBlockExpression();
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in iterator loop expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  it.setBody(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<IteratorLoopExpression>(it));
}

StringResult<std::shared_ptr<ast::Expression>>
Parser::parsePredicatePatternLoopExpression(std::span<ast::OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  PredicatePatternLoopExpression pat = {loc};

  if (!checkKeyWord(KeyWordKind::KW_WHILE))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse while keyword");
  assert(eatKeyWord(KeyWordKind::KW_WHILE));

  if (!checkKeyWord(KeyWordKind::KW_LET))
    return StringResult<std::shared_ptr<ast::Expression>>(
        "failed to parse let keyword");
  assert(eatKeyWord(KeyWordKind::KW_LET));

  StringResult<std::shared_ptr<ast::patterns::Pattern>> pred = parsePattern();
  if (!pred) {
    llvm::errs()
        << "failed to parse pattern in predicate pattern loop expression: "
        << pred.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  pat.setPattern(pred.getValue());

  StringResult<ast::Scrutinee> scrut = parseScrutinee();
  if (!scrut) {
    llvm::errs()
        << "failed to parse scrutinee in predicate pattern loop expression: "
        << scrut.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  pat.setScrutinee(scrut.getValue());

  StringResult<std::shared_ptr<ast::Expression>> block = parseBlockExpression();
  if (!block) {
    llvm::errs() << "failed to parse block expression in predicate pattern "
                    "loop expression: "
                 << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  pat.setBody(block.getValue());

  return StringResult<std::shared_ptr<ast::Expression>>(
      std::make_shared<PredicatePatternLoopExpression>(pat));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseInfiniteLoopExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  InfiniteLoopExpression infini = {loc};

  if (!checkKeyWord(KeyWordKind::KW_LOOP))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse loop keyword");
  assert(eatKeyWord(KeyWordKind::KW_LOOP));

  StringResult<std::shared_ptr<ast::Expression>> block = parseBlockExpression();
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in infinite loop expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  infini.setBody(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<InfiniteLoopExpression>(infini));
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parsePredicateLoopExpression(std::span<OuterAttribute>) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  PredicateLoopExpression pred = {loc};

  if (!checkKeyWord(KeyWordKind::KW_WHILE))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
        "failed to parse while keyword");
  assert(eatKeyWord(KeyWordKind::KW_WHILE));

  Restrictions restrictions;
  StringResult<std::shared_ptr<ast::Expression>> expr =
      parseExpression({}, restrictions);
  if (!expr) {
    llvm::errs() << "failed to parse expression in predicate loop expression: "
                 << expr.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  pred.setCondition(expr.getValue());

  StringResult<std::shared_ptr<ast::Expression>> block = parseBlockExpression();
  if (!block) {
    llvm::errs()
        << "failed to parse block expression in predicate loop expression: "
        << block.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }

  pred.setBody(block.getValue());

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<PredicateLoopExpression>(pred));
}

} // namespace rust_compiler::parser
