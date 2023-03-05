#include "AST/Types/BareFunctionType.h"

#include "ADT/Result.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkMaybeNamedParamLeadingComma() {
  if (!check(TokenKind::Comma))
    return false;

  assert(eat(TokenKind::Comma));

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in check maybe name "
                      "param leading comma: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    if (checkIdentifier() && check(TokenKind::Colon, 1))
      return true;
    if (check(TokenKind::Underscore) && check(TokenKind::Colon, 1))
      return true;
    if (check(TokenKind::Colon))
      return true;
    return false;
  }

  if (checkIdentifier() && check(TokenKind::Colon, 1))
    return true;
  if (check(TokenKind::Underscore) && check(TokenKind::Colon, 1))
    return true;
  if (check(TokenKind::Colon))
    return true;

  return false;
}

StringResult<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseMaybeNamedFunctionParameters() {
  Location loc = getLocation();
  MaybeNamedFunctionParameters maybe = {loc};

  StringResult<ast::types::MaybeNamedParam> namedParam = parseMaybeNamedParam();
  if (!namedParam) {
    llvm::errs() << "failed to parse mabe named param in parse maybe named "
                    "function parameters: "
                 << namedParam.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  maybe.addParameter(namedParam.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          "failed to parse maybe named function parameters: eof");
    } else if (check(TokenKind::ParenClose)) {
      // done
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          maybe);
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      // done
      assert(eat(TokenKind::Comma));
      maybe.setTrailingComma();
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          maybe);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      StringResult<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (!namedParam) {
        llvm::errs() << "failed to parse mabe named param in parse maybe named "
                        "function parameters: "
                     << namedParam.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      maybe.addParameter(namedParam.getValue());
    } else {
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          "failed to parse maybe named function parameters");
    }
  }
  return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
      "failed to parse maybe named function parameters: eof");
}

StringResult<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseMaybeNamedFunctionParametersVariadic() {
  Location loc = getLocation();
  MaybeNamedFunctionParametersVariadic maybe = {loc};

  StringResult<ast::types::MaybeNamedParam> namedParam = parseMaybeNamedParam();
  if (!namedParam) {
    llvm::errs() << "failed to parse mabe named param in parse maybe named "
                    "function parameters variadic: "
                 << namedParam.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  maybe.addParameter(namedParam.getValue());

  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          "failed to parse maybe named function parameters variadic: eof");
    } else if (check(TokenKind::DotDotDot) && check(TokenKind::ParenClose, 1)) {
      // done
      assert(eat(TokenKind::DotDotDot));
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          maybe);
    } else if (check(TokenKind::Comma) && checkOuterAttribute(1)) {
      assert(eat(TokenKind::Comma));
      StringResult<std::vector<ast::OuterAttribute>> outer =
          parseOuterAttributes();
      if (!outer) {
        llvm::errs()
            << "failed to parse outer paramaeters in parse maybe named "
               "function parameters variadic: "
            << outer.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      std::vector<ast::OuterAttribute> out = outer.getValue();
      maybe.setOuterAttributes(out);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      StringResult<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (!namedParam) {
        llvm::errs()
            << "failed to parse maybe named param in parse maybe named "
               "function parameters variadic: "
            << namedParam.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      maybe.addParameter(namedParam.getValue());
    } else {
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          "failed to parse maybe named function parameters variadic");
    }
  }
  return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
      "failed to parse maybe named function parameters: eof");
}

StringResult<ast::types::MaybeNamedParam> Parser::parseMaybeNamedParam() {
  Location loc = getLocation();

  MaybeNamedParam param = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (!outer) {
      llvm::errs() << "failed to parse outer attributes in parse maybe named "
                      "parameter: "
                   << outer.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> ot = outer.getValue();
    param.setOuterAttributes(ot);
  }

  if (check(TokenKind::Identifier)) {
    param.setIdentifier(getToken().getIdentifier());
    assert(eat(TokenKind::Identifier));
    if (!check(TokenKind::Colon))
      return StringResult<ast::types::MaybeNamedParam>(
          "failed to parse : in maybe named param");
    assert(eat(TokenKind::Colon));
  } else if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    param.setUnderscore();
    if (!check(TokenKind::Colon))
      return StringResult<ast::types::MaybeNamedParam>(
          "failed to parse : in maybe named param");
    assert(eat(TokenKind::Colon));
  }

  StringResult<std::shared_ptr<ast::types::TypeExpression>> type = parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in parse maybe named "
                    "parameter: "
                 << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  param.setType(type.getValue());

  return StringResult<ast::types::MaybeNamedParam>(param);
}

StringResult<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseFunctionParametersMaybeNamedVariadic() {
  CheckPoint cp = getCheckPoint();

  StringResult<ast::types::MaybeNamedParam> namedParam = parseMaybeNamedParam();
  if (!namedParam) {
    llvm::errs() << "failed to parse maybe named param  in function parameters "
                    "maybe varadic pattern: "
                 << namedParam.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  while (true) {
    if (check(TokenKind::Eof)) {
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          "failed to parse in function parameters maybe named variadic: "
          "eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      recover(cp);
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          parseMaybeNamedFunctionParameters());
    } else if (check(TokenKind::ParenClose)) {
      recover(cp);
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          parseMaybeNamedFunctionParameters());
    } else if (checkMaybeNamedParamLeadingComma()) {
      assert(eat(TokenKind::Comma));
      StringResult<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (!namedParam) {
        llvm::errs() << "failed to parse maybe named param  in function "
                        "parameters "
                        "maybe varadic: "
                     << namedParam.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
    } else if (check(TokenKind::Comma) && checkOuterAttribute(1)) {
      recover(cp);
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          parseMaybeNamedFunctionParametersVariadic());
    } else if (check(TokenKind::Comma) && check(TokenKind::DotDotDot, 1)) {
      recover(cp);
      return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
          parseMaybeNamedFunctionParametersVariadic());
    }
  }
  return StringResult<ast::types::FunctionParametersMaybeNamedVariadic>(
      "failed to parse in function parameters maybe named variadic");
}

StringResult<ast::types::BareFunctionReturnType>
Parser::parseBareFunctionReturnType() {
  Location loc = getLocation();

  BareFunctionReturnType qual = {loc};

  if (!check(TokenKind::RArrow))
    return StringResult<ast::types::BareFunctionReturnType>(
        "failed to parse -> in bare function return type");
  assert(eat(TokenKind::RArrow));

  StringResult<std::shared_ptr<ast::types::TypeExpression>> text =
      parseTypeNoBounds();
  if (!text) {
    llvm::errs()
        << "failed to parse type no bounds  in bare function return type "
           "maybe varadic: "
        << text.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  qual.setType(text.getValue());

  return StringResult<ast::types::BareFunctionReturnType>(qual);
}

StringResult<ast::types::FunctionTypeQualifiers>
Parser::parseFunctionTypeQualifiers() {
  Location loc = getLocation();

  FunctionTypeQualifiers qual = {loc};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
    qual.setUnsafe();
  }

  if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    assert(eatKeyWord(KeyWordKind::KW_EXTERN));

    StringResult<Abi> ab = parseAbi();
    if (!ab) {
      llvm::errs() << "failed to parse abi in parse "
                      "function type qualifiers"
                   << ab.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    qual.setAbi(ab.getValue());
  }

  return StringResult<ast::types::FunctionTypeQualifiers>(qual);
}

StringResult<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseBareFunctionType() {
  Location loc = getLocation();

  BareFunctionType bare = {loc};

  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    StringResult<ast::types::ForLifetimes> forL = parseForLifetimes();
    if (!forL) {
      llvm::errs() << "failed to parse for lifetimes in bare function type "
                   << forL.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    bare.setForLifetimes(forL.getValue());
  }

  StringResult<ast::types::FunctionTypeQualifiers> qual =
      parseFunctionTypeQualifiers();
  if (!qual) {
    llvm::errs()
        << "failed to parse function type qualifiers in bare function type "
        << qual.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  bare.setQualifiers(qual.getValue());

  if (!checkKeyWord(KeyWordKind::KW_FN))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse fn keywordk in bare function type");
  assert(eatKeyWord(KeyWordKind::KW_FN));

  if (!check(TokenKind::ParenOpen))
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        "failed to parse ( in bare function type");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    if (check((TokenKind::FatArrow))) {
      StringResult<ast::types::BareFunctionReturnType> ret =
          parseBareFunctionReturnType();
      if (!ret) {
        llvm::errs() << "failed to parse bare function return type in parse "
                        "bare function type"
                     << ret.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      bare.setReturnType(ret.getValue());
    }
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<BareFunctionType>(bare));
  } else {
    StringResult<FunctionParametersMaybeNamedVariadic> varadic =
        parseFunctionParametersMaybeNamedVariadic();
    if (!varadic) {
      llvm::errs() << "failed to parse function parameters maybe named "
                      "variadic in parse "
                      "bare function type"
                   << varadic.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    bare.setParameters(varadic.getValue());
    return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
        std::make_shared<BareFunctionType>(bare));
  }
  return StringResult<std::shared_ptr<ast::types::TypeExpression>>(
      "failed to parse bare function type");
}

} // namespace rust_compiler::parser
