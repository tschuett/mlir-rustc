#include "Parser/Parser.h"

#include <llvm/Support/raw_os_ostream.h>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

llvm::Expected<ast::SelfParam> Parser::parseSelfParam() {
  if(checkOuterAttribute()) {
  }
}

bool Parser::checkSelfParam() {
  if (check(TokenKind::And)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_MUT)) {
    return true;
  } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
    return true;
  }
  return false;
}

llvm::Expected<ast::FunctionParameters> Parser::parseFunctionParameters() {
  Location loc = getLocation();

  ast::FunctionParameters parameters = {loc};

  if (checkSelfParam()) {
    llvm::Expected<ast::SelfParam> selfParam = parseSelfParam();
    if (auto e = selfParam.takeError()) {
      llvm::errs() << "failed to parse self param: " << toString(std::move(e))
                   << "\n";
      exit(EXIT_FAILURE);
    }
    parameters.addSelfParam(*selfParam);
    if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
      assert(eat(TokenKind::Comma));
      return parameters;
    } else if (check(TokenKind::ParenClose)) {
      return parameters;
    } else {
      // continue
      while (true) {
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
        return parameters;
      } else if (check(TokenKind::Comma) && check(TokenKind::ParenClose, 1)) {
        assert(eat(TokenKind::Comma));
        return parameters;
      } else {
        llvm::Expected<ast::FunctionParam> param = parseFunctionParam();
        if (auto e = param.takeError()) {
          llvm::errs() << "failed to parse function param: "
                       << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        parameters.addFunctionParam(*param);
      }
    }
  }
  // FIXME
}

llvm::Expected<ast::FunctionParamPattern> Parser::parseFunctionParamPattern() {
  Location loc = getLocation();

  llvm::Expected<std::shared_ptr<ast::patterns::PatternNoTopAlt>> pattern =
      parsePatternNoTopAlt();

  if (auto e = pattern.takeError()) {
    llvm::errs() << "failed to parse pattern no top alt: "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (!check(TokenKind::Colon)) {
    // error
  }

  assert(eat(TokenKind::Colon));

  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    FunctionParamPattern pat = (loc);
    pat.setName(*pattern);
    return pat;
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type: " << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  FunctionParamPattern pat = (loc);
  pat.setName(*pattern);
  pat.setType(*type);
  return pat;
}

llvm::Expected<ast::FunctionParam> Parser::parseFunctionParam() {
  Location loc = getLocation();
  std::vector<ast::OuterAttribute> outerAttributes;

  if (checkOuterAttribute()) {
    llvm::Expected<std::vector<ast::OuterAttribute>> parsedOuterAttributes =
        parseOuterAttributes();
    if (auto e = parsedOuterAttributes.takeError()) {
      llvm::errs() << "failed to parse outer attributes: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    outerAttributes = *parsedOuterAttributes;
  }

  if (check(TokenKind::DotDotDot)) {
    // done
    assert(eat(TokenKind::DotDotDot));
    FunctionParam param = {loc, FunctionParamKind::DotDotDot};
    param.setAttributes(outerAttributes);
    return param;
  }

  // ???
}

// llvm::Expected<ast::FunctionSignature> Parser::parseFunctionsignature() {}

llvm::Expected<ast::FunctionQualifiers> Parser::parseFunctionQualifiers() {}

llvm::Expected<std::shared_ptr<ast::VisItem>>
Parser::parseFunction(std::optional<ast::Visibility> vis) {
  Location loc = getLocation();

  Function fun = {loc, vis};

  if (checkKeyWord(KeyWordKind::KW_CONST) ||
      checkKeyWord(KeyWordKind::KW_ASYNC) ||
      checkKeyWord(KeyWordKind::KW_UNSAFE) ||
      checkKeyWord(KeyWordKind::KW_EXTERN)) {
    llvm::Expected<ast::FunctionQualifiers> qualifiers =
        parseFunctionQualifiers();
    if (auto e = qualifiers.takeError()) {
      llvm::errs() << "failed to parse function qualifiers: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fun.setQualifiers(*qualifiers);
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

  if (check(TokenKind::Lt)) {
    llvm::Expected<ast::GenericParams> genericParams =
        parseGenericParams();
    if (auto e = genericParams.takeError()) {
      llvm::errs() << "failed to parse generic parameters: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fun.setGenericParams(*genericParams);
  }

  if (!check(TokenKind::ParenOpen)) {
    // error
  }

  assert(eat(TokenKind::ParenOpen));

  // optional
  if (!check(TokenKind::ParenClose)) {
    llvm::Expected<ast::FunctionParameters> parameters =
        parseFunctionParameters();
    if (auto e = parameters.takeError()) {
      llvm::errs() << "failed to parse fn parameters: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fun.setParameters(*parameters);
  }

  assert(eat(TokenKind::ParenClose));

  // return type
  if (check(TokenKind::RArrow)) {
    assert(eat(TokenKind::RArrow));
    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> returnType =
        parseType();
    if (auto e = returnType.takeError()) {
      llvm::errs() << "failed to parse fn return type: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fun.setReturnType(*returnType);
  }

  if (checkKeyWord(KeyWordKind::KW_WHERE)) {
    llvm::Expected<ast::WhereClause> whereClause =
        parseWhereClause();
    if (auto e = whereClause.takeError()) {
      llvm::errs() << "failed to parse fn where clause: "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    fun.setWhereClasue(*whereClause);
  }

  if (check(TokenKind::Semi)) {
    assert(eat(TokenKind::Semi));
    return std::make_shared<ast::VisItem>(fun);
  }

  llvm::Expected<std::shared_ptr<ast::BlockExpression>> body =
      parseBlockExpression();
  if (auto e = body.takeError()) {
    llvm::errs() << "failed to parse fn bofy: " << toString(std::move(e))
                 << "\n";
    exit(EXIT_FAILURE);
  }
  fun.setBody(*body);

  return std::make_shared<ast::VisItem>(fun);
}

} // namespace rust_compiler::parser

