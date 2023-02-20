#include "AST/InfiniteLoopExpression.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::Expression>> Parser::parseLoopExpression() {
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseIteratorLoopExpression() {
  Location loc = getLocation();

  IteratorLoopExpression it = {loc};

  if (!checkKeyWord(KeyWordKind::KW_FOR))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse for keyword");
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pred = parsePattern();
  if (auto e = pred.takeError()) {
    llvm::errs() << "failed to parse pattern in iterator loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  it.setPattern(*pred);

  if (!checkKeyWord(KeyWordKind::KW_IN))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse in keyword");
  assert(eatKeyWord(KeyWordKind::KW_IN));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in iterator loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  it.setExpression(*expr);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block expression in iterator loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  it.setBody(*block);

  return std::make_shared<IteratorLoopExpression>(it);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parsePredicatePatternLoopExpression() {
  Location loc = getLocation();

  PredicatePatternLoopExpression pat = {loc};

  if (!checkKeyWord(KeyWordKind::KW_WHILE))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse while keyword");
  assert(eatKeyWord(KeyWordKind::KW_WHILE));

  if (!checkKeyWord(KeyWordKind::KW_LET))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse let keyword");
  assert(eatKeyWord(KeyWordKind::KW_LET));

  llvm::Expected<std::shared_ptr<ast::patterns::Pattern>> pred = parsePattern();
  if (auto e = pred.takeError()) {
    llvm::errs() << "failed to parse pattern in predicate pattern loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  pat.setPattern(*pred);

  llvm::Expected<ast::Scrutinee> scrut = parseScrutinee();
  if (auto e = scrut.takeError()) {
    llvm::errs() << "failed to parse scrutinee in predicate pattern loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  pat.setScrutinee(*scrut);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs()
        << "failed to parse block expression in predicate pattern loop: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  pat.setBody(*block);

  return std::make_shared<PredicatePatternLoopExpression>(pat);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parseInfiniteLoopExpression() {
  Location loc = getLocation();

  InfiniteLoopExpression infini = {loc};

  if (!checkKeyWord(KeyWordKind::KW_LOOP))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse loop keyword");
  assert(eatKeyWord(KeyWordKind::KW_LOOP));

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block expression in infinite loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  infini.setBody(*block);

  return std::make_shared<InfiniteLoopExpression>(infini);
}

llvm::Expected<std::shared_ptr<ast::Expression>>
Parser::parsePredicateLoopExpression() {
  Location loc = getLocation();

  PredicateLoopExpression pred = {loc};

  if (!checkKeyWord(KeyWordKind::KW_WHILE))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse while keyword");
  assert(eatKeyWord(KeyWordKind::KW_WHILE));

  llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
  if (auto e = expr.takeError()) {
    llvm::errs() << "failed to parse expression in predicate loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  pred.setCondition(*expr);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs() << "failed to parse block expression in predicate loop: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  pred.setBody(*block);

  return std::make_shared<PredicateLoopExpression>(pred);
}

} // namespace rust_compiler::parser
