#include "AST/BlockExpression.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionReturnType.h"
#include "AST/SelfParam.h"
#include "AST/ShorthandSelf.h"
#include "AST/TypedSelf.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_os_ostream.h>
#include <memory>
#include <optional>

using namespace llvm;
using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::parser {

StringResult<std::shared_ptr<ast::SelfParam>> Parser::parseShorthandSelf() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  ShorthandSelf self = {loc};

  // FIXME Lifetime

  if (check(TokenKind::And)) {
    assert(eat(TokenKind::And));
    self.setAnd();
  }

  if (checkKeyWord(KeyWordKind::KW_MUT)) {
    assert(eatKeyWord(KeyWordKind::KW_MUT));
    self.setMut();
  }

  if (!checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return StringResult<std::shared_ptr<ast::SelfParam>>(
        "failed to parse self");

  assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));

  return StringResult<std::shared_ptr<ast::SelfParam>>(
      std::make_shared<ShorthandSelf>(self));
}

StringResult<std::shared_ptr<ast::SelfParam>> Parser::parseTypedSelf() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  TypedSelf self = {loc};

  if (checkKeyWord(KeyWordKind::KW_MUT))
    self.setMut();

  if (!checkKeyWord(KeyWordKind::KW_SELFVALUE))
    return StringResult<std::shared_ptr<ast::SelfParam>>(
        "failed to parse self");

  assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));

  if (!check(TokenKind::Colon))
    return StringResult<std::shared_ptr<ast::SelfParam>>(
        "failed to parse colon");

  assert(eat(TokenKind::Colon));

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    llvm::errs() << "failed to parse type: " << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  self.setType(type.getValue());

  return StringResult<std::shared_ptr<ast::SelfParam>>(
      std::make_shared<TypedSelf>(self));
}

StringResult<ast::SelfParam> Parser::parseSelfParam() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  llvm::errs() << "parseSelfParam" << "\n";

  SelfParam self = {loc};

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> parsedOuterAttributes =
        parseOuterAttributes();
    if (!parsedOuterAttributes) {
      llvm::errs() << "failed to parse outer attributes in self param: "
                   << parsedOuterAttributes.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    std::vector<ast::OuterAttribute> outer = parsedOuterAttributes.getValue();
    self.setOuterAttributes(outer);
  }

  if (check(TokenKind::And)) {
    // ShordhandSelf
    StringResult<std::shared_ptr<ast::SelfParam>> shortA = parseShorthandSelf();
    if (!shortA) {
      llvm::errs() << "failed to parse shorthand self in self param: "
                   << shortA.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    self.setSelf(SelfParamKind::ShorthandSelf, shortA.getValue());
  } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
    if (checkKeyWord(KeyWordKind::KW_SELFVALUE, 1)) {
      if (check(TokenKind::Colon, 2)) {
        StringResult<std::shared_ptr<ast::SelfParam>> shortA = parseTypedSelf();
        if (!shortA) {
          llvm::errs() << "failed to parse shortand self in self param: "
                       << shortA.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        self.setSelf(SelfParamKind::TypeSelf, shortA.getValue());
        // TypedSelf
      } else {
        StringResult<std::shared_ptr<ast::SelfParam>> shortA =
            parseShorthandSelf();
        if (!shortA) {
          llvm::errs() << "failed to parse shortand self in self param: "
                       << shortA.getError() << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        self.setSelf(SelfParamKind::ShorthandSelf, shortA.getValue());
      }
    }
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    if (check(TokenKind::Colon, 1)) {
      StringResult<std::shared_ptr<ast::SelfParam>> shortA = parseTypedSelf();
      if (!shortA) {
        llvm::errs() << "failed to parse shortand self in self param: "
                     << shortA.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      self.setSelf(SelfParamKind::TypeSelf, shortA.getValue());
    } else {
      StringResult<std::shared_ptr<ast::SelfParam>> shortA =
          parseShorthandSelf();
      if (!shortA) {
        llvm::errs() << "failed to parse shortand self in self param: "
                     << shortA.getError() << "\n";
        printFunctionStack();
        exit(EXIT_FAILURE);
      }
      self.setSelf(SelfParamKind::ShorthandSelf, shortA.getValue());
    }
  }
  return StringResult<ast::SelfParam>("failed to parse colonself param");
}

bool Parser::checkSelfParam() {
  CheckPoint cp = getCheckPoint();

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> parsedOuterAttributes =
        parseOuterAttributes();
    if (!parsedOuterAttributes) {
      llvm::errs() << "failed to parse outer attributes in check self param: "
                   << parsedOuterAttributes.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv(
              "{0} {1}",
              "failed to parse outer attributes in check self param: ",
              parsedOuterAttributes.getError())
              .str();
      return StringResult<ast::FunctionParam>(s);
    }
  }

  if (check(TokenKind::And)) {
    recover(cp);
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
    recover(cp);
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    recover(cp);
    return true;
  }

  recover(cp);
  return false;
}

StringResult<ast::FunctionParameters> Parser::parseFunctionParameters() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  ast::FunctionParameters parameters = {loc};

  if (checkSelfParam()) {
    StringResult<ast::SelfParam> selfParam = parseSelfParam();
    if (!selfParam) {
      llvm::errs() << "failed to parse self param: " << selfParam.getError()
                   << "\n";
      printFunctionStack();
      // exit(EXIT_FAILURE);
      std::string s =
          llvm::formatv("{0}\n{1}",
                        "failed to parse self param: ", selfParam.getError())
              .str();
      return StringResult<ast::FunctionParameters>(s);
    }
    parameters.addSelfParam(selfParam.getValue());
    if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return StringResult<ast::FunctionParameters>(parameters);
    } else if (check(TokenKind::ParenClose)) {
      return StringResult<ast::FunctionParameters>(parameters);
    } else {
      while (true) {
        // copy and paste
        if (check(TokenKind::Eof)) {
          llvm::errs() << "unexpected EOF"
                       << "\n";
          exit(EXIT_FAILURE);
        } else if (check(TokenKind::ParenClose)) {
          return StringResult<ast::FunctionParameters>(parameters);
        } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
          assert(eat(TokenKind::Comma));
          return StringResult<ast::FunctionParameters>(parameters);
        } else if (check(TokenKind::Comma)) {
          assert(eat(TokenKind::Comma));
        } else {
          StringResult<ast::FunctionParam> param = parseFunctionParam();
          if (!param) {
            llvm::errs() << "failed to parse function param: "
                         << param.getError() << "\n";
            printFunctionStack();
            exit(EXIT_FAILURE);
          }
          parameters.addFunctionParam(param.getValue());
        }
      }
    }
  } else {
    // no self param
    while (true) {
      if (check(TokenKind::Eof)) {
        llvm::errs() << "unexpected EOF"
                     << "\n";
        exit(EXIT_FAILURE);
      } else if (check(TokenKind::ParenClose)) {
        return StringResult<ast::FunctionParameters>(parameters);
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        return StringResult<ast::FunctionParameters>(parameters);
      } else if (check(TokenKind::Comma)) {
        assert(eat(TokenKind::Comma));
      } else {
        StringResult<ast::FunctionParam> param = parseFunctionParam();
        if (!param) {
          llvm::errs() << "failed to parse function param: " << param.getError()
                       << "\n";
          printFunctionStack();
          exit(EXIT_FAILURE);
        }
        parameters.addFunctionParam(param.getValue());
      }
    }
  }
  // FIXME
}

StringResult<ast::FunctionParamPattern> Parser::parseFunctionParamPattern() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  //  llvm::errs() << "parseFunctionParamPattern"
  //               << "\n";
  //
  //  llvm::errs() << "parseFunctionParamPattern: pattern"
  //               << "\n";

  StringResult<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();
  if (!pattern) {
    llvm::errs()
        << "failed to parse pattern no top alt in function param pattern: "
        << pattern.getError() << "\n";
    std::string s =
        llvm::formatv(
            "{0} {1}",
            "failed to parse pattern no top alt in function param pattern: ",
            pattern.getError())
            .str();
    printFunctionStack();
    return StringResult<ast::FunctionParamPattern>(s);
  }
  if (!check(TokenKind::Colon)) {
    // error
    llvm::errs() << "failed to parse : token in function param pattern: "
                 << Token2String(getToken().getKind()) << "\n";
    std::string s = "failed to parse : token in function param pattern";
    return StringResult<ast::FunctionParamPattern>(s);
  }

  assert(eat(TokenKind::Colon));

  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    FunctionParamPattern pat = (loc);
    pat.setName(pattern.getValue());
    return StringResult<ast::FunctionParamPattern>(pat);
  }

  //  llvm::errs() << "parseFunctionParamPattern: type"
  //               << "\n";

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    llvm::errs() << "failed to parse type in function param pattern: "
                 << type.getError() << "\n";
    printFunctionStack();
    // exit(EXIT_FAILURE);
    std::string s =
        llvm::formatv(
            "{0}\n{1}",
            "failed to parse type in function param pattern: ", type.getError())
            .str();
    return StringResult<ast::FunctionParamPattern>(s);
  }

  //  llvm::errs() << "parseFunctionParamPattern: done: "
  //               << Token2String(getToken().getKind()) << "\n";

  FunctionParamPattern pat = (loc);
  pat.setName(pattern.getValue());
  pat.setType(type.getValue());
  return StringResult<ast::FunctionParamPattern>(pat);
}

StringResult<ast::FunctionParam> Parser::parseFunctionParam() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  std::vector<ast::OuterAttribute> outerAttributes;

  if (checkOuterAttribute()) {
    StringResult<std::vector<ast::OuterAttribute>> parsedOuterAttributes =
        parseOuterAttributes();
    if (!parsedOuterAttributes) {
      llvm::errs() << "failed to parse outer attributes in function param: "
                   << parsedOuterAttributes.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}",
                        "failed to parse outer attributes in function param: ",
                        parsedOuterAttributes.getError())
              .str();
      // exit(EXIT_FAILURE);
      return StringResult<ast::FunctionParam>(s);
    }
    outerAttributes = parsedOuterAttributes.getValue();
  }

  // FIXME ignore naked type
  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    FunctionParam param = {loc, FunctionParamKind::DotDotDot};
    param.setOuterAttributes(outerAttributes);
    return StringResult<ast::FunctionParam>(param);
  } else {
    FunctionParam param = {loc, FunctionParamKind::Pattern};
    StringResult<ast::FunctionParamPattern> pattern =
        parseFunctionParamPattern();
    if (!pattern) {
      llvm::errs() << "failed to parse pattern in function param: "
                   << pattern.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv(
              "{0} {1}",
              "failed to parse pattern in function param: ", pattern.getError())
              .str();
      // exit(EXIT_FAILURE);
      return StringResult<ast::FunctionParam>(s);
    }
    param.setPattern(pattern.getValue());
    param.setOuterAttributes(outerAttributes);
    return StringResult<ast::FunctionParam>(param);
  }
  return StringResult<ast::FunctionParam>("failed to parse function param");
}

StringResult<ast::Abi> Parser::parseAbi() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  Abi abi = {loc};

  if (check(TokenKind::STRING_LITERAL)) {
    // FIXME
    return StringResult<ast::Abi>(abi);
  } else if (check(TokenKind::RAW_STRING_LITERAL)) {
    // FIXME
    return StringResult<ast::Abi>(abi);
  }

  return StringResult<ast::Abi>("failed to parse Abi");
}

StringResult<ast::FunctionQualifiers> Parser::parseFunctionQualifiers() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  FunctionQualifiers qual = {loc};

  if (checkKeyWord(KeyWordKind::KW_CONST)) {
    assert(eatKeyWord(KeyWordKind::KW_CONST));
    qual.setConst();
  }

  if (checkKeyWord(KeyWordKind::KW_ASYNC)) {
    assert(eatKeyWord(KeyWordKind::KW_ASYNC));
    qual.setAsync();
  }

  if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    assert(eatKeyWord(KeyWordKind::KW_UNSAFE));
    qual.setUnsafe();
  }

  if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    assert(eatKeyWord(KeyWordKind::KW_EXTERN));

    StringResult<ast::Abi> abi = Parser::parseAbi();
    if (!abi) {
      llvm::errs() << "failed to parse abi in function qualifiers: "
                   << abi.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    qual.setAbi(abi.getValue());
  }

  return StringResult<ast::FunctionQualifiers>(qual);
}

StringResult<ast::FunctionReturnType> Parser::parseFunctionReturnType() {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();
  FunctionReturnType t = {loc};

  if (!check(TokenKind::RArrow)) {
    return StringResult<ast::FunctionReturnType>(
        "failed to parse -> token in function return type");
  }
  assert(eat(TokenKind::RArrow));

  Result<std::shared_ptr<ast::types::TypeExpression>, std::string> type =
      parseType();
  if (!type) {
    llvm::errs() << "failed to parse type: " << type.getError() << "\n";
    printFunctionStack();
    exit(EXIT_FAILURE);
  }
  t.setType(type.getValue());

  return StringResult<ast::FunctionReturnType>(t);
}

StringResult<std::shared_ptr<ast::Item>>
Parser::parseFunction(std::optional<ast::Visibility> vis) {
  ParserErrorStack raai = {this, __PRETTY_FUNCTION__};
  Location loc = getLocation();

  Function fun = {loc, vis};

  //  llvm::outs() << "parseFunction"
  //               << "\n";

  if (checkKeyWord(KeyWordKind::KW_CONST) ||
      checkKeyWord(KeyWordKind::KW_ASYNC) ||
      checkKeyWord(KeyWordKind::KW_UNSAFE) ||
      checkKeyWord(KeyWordKind::KW_EXTERN)) {
    StringResult<ast::FunctionQualifiers> qualifiers =
        parseFunctionQualifiers();
    if (!qualifiers) {
      llvm::errs() << "failed to parse function qualifiers in function: "
                   << qualifiers.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fun.setQualifiers(qualifiers.getValue());
  }

  if (!checkKeyWord(KeyWordKind::KW_FN)) {
    llvm::errs() << "found no fn"
                 << "\n";
    exit(EXIT_FAILURE);
  }

  assert(eatKeyWord(KeyWordKind::KW_FN));

  if (!check(TokenKind::Identifier)) {
    llvm::errs() << "found no fn identifier"
                 << "\n";
  }

  Token id = getToken();
  std::string identifier = id.getIdentifier();

  assert(eat(TokenKind::Identifier));

  fun.setIdentifier(identifier);

  if (check(TokenKind::Lt)) {
    StringResult<ast::GenericParams> genericParams = parseGenericParams();
    if (!genericParams) {
      llvm::errs() << "failed to parse generic params in function: "
                   << genericParams.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fun.setGenericParams(genericParams.getValue());
  }

  if (!check(TokenKind::ParenOpen)) {
    return StringResult<std::shared_ptr<ast::Item>>(
        "failed to parse generic params");
  }

  assert(eat(TokenKind::ParenOpen));

  // optional
  if (!check(TokenKind::ParenClose)) {
    StringResult<ast::FunctionParameters> parameters =
        parseFunctionParameters();
    if (!parameters) {
      llvm::errs() << "failed to parse function parameters in function: "
                   << parameters.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fun.setParameters(parameters.getValue());
  }

  assert(eat(TokenKind::ParenClose));

  // return type
  if (check(TokenKind::RArrow)) {
    StringResult<FunctionReturnType> returnType = parseFunctionReturnType();
    if (!returnType) {
      llvm::errs() << "failed to parse function return type in function: "
                   << returnType.getError() << "\n";
      printFunctionStack();
      exit(EXIT_FAILURE);
    }
    fun.setReturnType(returnType.getValue());
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    StringResult<ast::WhereClause> whereClause = parseWhereClause();
    if (!whereClause) {
      llvm::errs() << "failed to parse where clause in function: "
                   << whereClause.getError() << "\n";
      printFunctionStack();
      std::string s =
          llvm::formatv("{0} {1}", "failed to parse where clause in function",
                        whereClause.getError())
              .str();
      return StringResult<std::shared_ptr<ast::Item>>(s);
    }
    fun.setWhereClasue(whereClause.getValue());
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return StringResult<std::shared_ptr<ast::Item>>(
        std::make_shared<ast::VisItem>(fun));
  }

  //  llvm::outs() << "parseFunction: parse body"
  //               << "\n";

  Result<std::shared_ptr<ast::Expression>, std::string> body =
      parseBlockExpression({});
  if (!body) {
    llvm::errs() << "failed to parse body in function: " << body.getError()
                 << "\n";
    printFunctionStack();
    std::string s = llvm::formatv("{0} {1}", "failed to parse body in function",
                                  body.getError())
                        .str();
    return StringResult<std::shared_ptr<ast::Item>>(s);
  }
  fun.setBody(body.getValue());

  //  llvm::errs() << "function body: "
  //               << std::static_pointer_cast<BlockExpression>(body.getValue())
  //                      ->getExpressions()
  //                      .getSize()
  //               << "\n";

  return StringResult<std::shared_ptr<ast::Item>>(
      std::make_shared<Function>(fun));
}

} // namespace rust_compiler::parser
