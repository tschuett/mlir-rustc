#include "AST/Types/BareFunctionType.h"

#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace llvm;

namespace rust_compiler::parser {

bool Parser::checkMaybeNamedParamLeadingComma() {
  if (!check(TokenKind::Comma))
    return false;

  assert(eat(TokenKind::Comma));

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs()
          << "failed to parse outer attributes in maybe named function "
             "parameters "
             "maybe named variadic : "
          << toString(std::move(e)) << "\n";
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

llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseMaybeNamedFunctionParameters() {
  Location loc = getLocation();
  MaybeNamedFunctionParameters maybe = {loc};

  llvm::Expected<ast::types::MaybeNamedParam> namedParam =
      parseMaybeNamedParam();
  if (auto e = namedParam.takeError()) {
    llvm::errs() << "failed to parse maybe named param in maybe named function "
                    "parameters "
                    "maybe named variadic : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  maybe.addParameter(*namedParam);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse maybe named function parameters: eof");
    } else if (check(TokenKind::ParenClose)) {
      // done
      return maybe;
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      // done
      assert(eat(TokenKind::Comma));
      maybe.setTrailingComma();
      return maybe;
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (auto e = namedParam.takeError()) {
        llvm::errs()
            << "failed to parse maybe named param in maybe named function "
               "parameters "
               "maybe named variadic : "
            << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      maybe.addParameter(*namedParam);
    } else {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse maybe named function parameters");
    }
  }
  return createStringError(
      inconvertibleErrorCode(),
      "failed to parse maybe named function parameters: eof");
}

llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseMaybeNamedFunctionParametersVariadic() {
  Location loc = getLocation();
  MaybeNamedFunctionParametersVariadic maybe = {loc};

  llvm::Expected<ast::types::MaybeNamedParam> namedParam =
      parseMaybeNamedParam();
  if (auto e = namedParam.takeError()) {
    llvm::errs() << "failed to parse maybe named param in maybe named function "
                    "parameters "
                    "maybe named variadic : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  maybe.addParameter(*namedParam);

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse maybe named function parameters variadic: eof");
    } else if (check(TokenKind::DotDotDot) && check(TokenKind::ParenClose, 1)) {
      // done
      assert(eat(TokenKind::DotDotDot));
      return maybe;
    } else if (check(TokenKind::Comma) && checkOuterAttribute(1)) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<std::vector<ast::OuterAttribute>> outer =
          parseOuterAttributes();
      if (auto e = outer.takeError()) {
        llvm::errs()
            << "failed to parse outer attributes in maybe named function "
               "parameters "
               "maybe named variadic : "
            << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      maybe.setOuterAttributes(*outer);
    } else if (check(TokenKind::Comma)) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (auto e = namedParam.takeError()) {
        llvm::errs()
            << "failed to parse maybe named param in maybe named function "
               "parameters "
               "maybe named variadic : "
            << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      maybe.addParameter(*namedParam);
    } else {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse maybe named function parameters variadic");
    }
  }
  return createStringError(
      inconvertibleErrorCode(),
      "failed to parse maybe named function parameters: eof");
}

llvm::Expected<ast::types::MaybeNamedParam> Parser::parseMaybeNamedParam() {
  Location loc = getLocation();

  MaybeNamedParam param = {loc};

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> outer =
        parseOuterAttributes();
    if (auto e = outer.takeError()) {
      llvm::errs() << "failed to parse outer attributes in maybe named param : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    param.setOuterAttributes(*outer);
  }

  if (check(TokenKind::Identifier)) {
    param.setIdentifier(getToken().getIdentifier());
    assert(eat(TokenKind::Identifier));
    if (!check(TokenKind::Colon))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse : in maybe named param");
    assert(eat(TokenKind::Colon));
  } else if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    param.setUnderscore();
    if (!check(TokenKind::Colon))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse : in maybe named param");
    assert(eat(TokenKind::Colon));
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in maybe named param : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  param.setType(*type);

  return param;
}

llvm::Expected<ast::types::FunctionParametersMaybeNamedVariadic>
Parser::parseFunctionParametersMaybeNamedVariadic() {
  CheckPoint cp = getCheckPoint();

  llvm::Expected<ast::types::MaybeNamedParam> namedParam =
      parseMaybeNamedParam();
  if (auto e = namedParam.takeError()) {
    llvm::errs() << "failed to parse maybe named param in function parameters "
                    "maybe named variadic : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  while (true) {
    if (check(TokenKind::Eof)) {
      return createStringError(
          inconvertibleErrorCode(),
          "failed to parse in function parameters maybe named variadic: eof");
    } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      recover(cp);
      return parseMaybeNamedFunctionParameters();
    } else if (check(TokenKind::ParenClose)) {
      recover(cp);
      return parseMaybeNamedFunctionParameters();
    } else if (checkMaybeNamedParamLeadingComma()) {
      assert(eat(TokenKind::Comma));
      llvm::Expected<ast::types::MaybeNamedParam> namedParam =
          parseMaybeNamedParam();
      if (auto e = namedParam.takeError()) {
        llvm::errs()
            << "failed to parse maybe named param in function parameters "
               "maybe named variadic : "
            << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
    } else if (check(TokenKind::Comma) && checkOuterAttribute(1)) {
      recover(cp);
      return parseMaybeNamedFunctionParametersVariadic();
    } else if (check(TokenKind::Comma) && check(TokenKind::DotDotDot, 1)) {
      recover(cp);
      return parseMaybeNamedFunctionParametersVariadic();
    }
  }
  return createStringError(
      inconvertibleErrorCode(),
      "failed to parse in function parameters maybe named variadic");
}

llvm::Expected<ast::types::BareFunctionReturnType>
Parser::parseBareFunctionReturnType() {
  Location loc = getLocation();

  BareFunctionReturnType qual = {loc};

  if (!check(TokenKind::RArrow))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse -> in bare function return type");
  assert(eat(TokenKind::RArrow));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> text =
      parseTypeNoBounds();
  if (auto e = text.takeError()) {
    llvm::errs() << "failed to parse type no bounds in bare function type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  qual.setType(*text);

  return qual;
}

llvm::Expected<ast::types::FunctionTypeQualifiers>
Parser::parseFunctionTypeQualifiers() {
  Location loc = getLocation();

  FunctionTypeQualifiers qual = {loc};

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
    qual.setUnsafe();
  }

  if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    assert(eatKeyWord(KeyWordKind::KW_EXTERN));

    llvm::Expected<Abi> ab = parseAbi();
    if (auto e = ab.takeError()) {
      llvm::errs() << "failed to parse Abi in bare function type : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    qual.setAbi(*ab);
  }

  return qual;
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseBareFunctionType() {
  Location loc = getLocation();

  BareFunctionType bare = {loc};

  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
    if (auto e = forL.takeError()) {
      llvm::errs() << "failed to parse for lifetimes in bare function type : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    bare.setForLifetimes(*forL);
  }

  llvm::Expected<ast::types::FunctionTypeQualifiers> qual =
      parseFunctionTypeQualifiers();
  if (auto e = qual.takeError()) {
    llvm::errs() << "failed to parse for function type qualifiers in bare "
                    "function type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bare.setQualifiers(*qual);

  if (!checkKeyWord(KeyWordKind::KW_FN))
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse fn keywordk in bare function type");
  assert(eatKeyWord(KeyWordKind::KW_FN));

  if (!check(TokenKind::ParenOpen))
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse ( in bare function type");
  assert(eat(TokenKind::ParenOpen));

  if (check(TokenKind::ParenClose)) {
    assert(eat(TokenKind::ParenClose));
    if (check((TokenKind::FatArrow))) {
      llvm::Expected<ast::types::BareFunctionReturnType> ret =
          parseBareFunctionReturnType();
      if (auto e = ret.takeError()) {
        llvm::errs() << "failed to parse for return type in bare "
                        "function type : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      bare.setReturnType(*ret);
    }
    return std::make_shared<BareFunctionType>(bare);
  } else {
    llvm::Expected<FunctionParametersMaybeNamedVariadic> varadic =
        parseFunctionParametersMaybeNamedVariadic();
    if (auto e = varadic.takeError()) {
      llvm::errs() << "failed to parse function parameters maybe named "
                      "variadic  in bare "
                      "function type : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    bare.setParameters(*varadic);
    return std::make_shared<BareFunctionType>(bare);
  }
  return createStringError(inconvertibleErrorCode(),
                           "failed to parse bare function type");
}

} // namespace rust_compiler::parser
