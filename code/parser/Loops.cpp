#include "AST/InfiniteLoopExpression.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LabelBlockExpression.h"
#include "AST/OuterAttribute.h"
#include "AST/OuterAttributes.h"
#include "AST/PredicateLoopExpression.h"
#include "AST/PredicatePatternLoopExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

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
  Location loc = getLocation();
  LabelBlockExpression bloc = {loc};

  if (label)
    bloc.setLabel(*label);

  llvm::Expected<std::shared_ptr<ast::Expression>> block =
      parseBlockExpression();
  if (auto e = block.takeError()) {
    llvm::errs()
        << "failed to parse block expression in label block expression: "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bloc.setBlock(*block);

  return std::make_shared<LabelBlockExpression>(bloc);
}

// Result<LoopLabel, std::string> Parser::parseLoopLabel() {}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseLoopExpression(std::span<OuterAttribute>) {
  if (checkLoopLabel()) {
    if (check(TokenKind::LIFETIME_OR_LABEL) && check(TokenKind::Colon, 1)) {
      std::string label = getToken().getStorage();
      assert(eat(TokenKind::LIFETIME_OR_LABEL));
      assert(eat(TokenKind::Colon));

      if (checkKeyWord(KeyWordKind::KW_LOOP)) {
        return parseInfiniteLoopExpression(label);
      } else if (checkKeyWord(KeyWordKind::KW_WHILE) &&
                 checkKeyWord(KeyWordKind::KW_LET, 1)) {
        return parsePredicatePatternLoopExpression(label);
      } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
        return parsePredicateLoopExpression(label);
      } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
        return parseIteratorLoopExpression(label);
      } else {
        return parseLabelBlockExpression(label);
      }
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse loop expression with loop label");
    }
  }

  if (checkKeyWord(KeyWordKind::KW_LOOP)) {
    return parseInfiniteLoopExpression(std::nullopt);
  } else if (checkKeyWord(KeyWordKind::KW_WHILE) &&
             checkKeyWord(KeyWordKind::KW_LET, 1)) {
    return parsePredicatePatternLoopExpression(std::nullopt);
  } else if (checkKeyWord(KeyWordKind::KW_WHILE)) {
    return parsePredicateLoopExpression(std::nullopt);
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    return parseIteratorLoopExpression(std::nullopt);
  } else {
    return parseLabelBlockExpression(std::nullopt);
  }
  return createStringError(
      inconvertibleErrorCode(),
      "failed to parse loop expression without loop label");
}

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseIteratorLoopExpression(std::span<OuterAttribute>) {
  Location loc = getLocation();

  IteratorLoopExpression it = {loc};
  if (label)
    it.setLabel(*label);

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
Parser::parsePredicatePatternLoopExpression(std::optional<std::string> label) {
  Location loc = getLocation();

  PredicatePatternLoopExpression pat = {loc};
  if (label)
    pat.setLabel(*label);

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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parseInfiniteLoopExpression(std::optional < std::span<OuterAttribute>) {
  Location loc = getLocation();

  InfiniteLoopExpression infini = {loc};

  if (label)
    infini.setLabel(*label);

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

Result<std::shared_ptr<ast::Expression>, std::string>
Parser::parsePredicateLoopExpression(std::span<OuterAttribute>) {
  Location loc = getLocation();

  PredicateLoopExpression pred = {loc};
  if (label)
    pred.setLabel(*label);

  if (!checkKeyWord(KeyWordKind::KW_WHILE))
    return Result<std::shared_ptr<ast::Expression>, std::string>(
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

  return Result<std::shared_ptr<ast::Expression>, std::string>(
      std::make_shared<PredicateLoopExpression>(pred));
}

} // namespace rust_compiler::parser
