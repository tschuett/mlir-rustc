#include "AST/Types/ArrayType.h"
#include "AST/Types/InferredType.h"
#include "AST/Types/NeverType.h"
#include "AST/Types/SliceType.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<ast::types::ForLifetimes> Parser::parseForLifetimes() {
  Location loc = getLocation();

  ForLifetimes forL = {loc};

  if (!checkKeyWord(KeyWordKind::KW_FOR)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse for keyword in for lifetimes");
  }
  assert(eatKeyWord(KeyWordKind::KW_FOR));

  llvm::Expected<ast::GenericParams> genericParams = parseGenericParams();
  if (auto e = genericParams.takeError()) {
    llvm::errs() << "failed to parse generic params in for lifetimes : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  forL.setGenericParams(*genericParams);

  return forL;
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplType() {
  Location loc = getLocation();

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse impl keyword in impl trait");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  // FIXME
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectType() {
  Location loc = getLocation();

  bool dyn = false;

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    assert(eatKeyWord(KeyWordKind::KW_DYN));
    dyn = true;
  }

  // FIXME
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseArrayOrSliceType() {
  Location loc = getLocation();

  if (!check(TokenKind::SquareOpen)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse array or slice type");
  }

  assert(eat(TokenKind::SquareOpen));

  llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> type =
      parseType();
  if (auto e = type.takeError()) {
    llvm::errs() << "failed to parse type in array or slice type : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }

  if (check(TokenKind::SquareClose)) {
    // slice type
    SliceType slice = {loc};
    slice.setType(*type);

    return std::make_shared<SliceType>(slice);
  } else if (check(TokenKind::Semi)) {
    // array type
    // slice type
    ArrayType arr = {loc};
    arr.setType(*type);

    assert(eat(TokenKind::Semi));

    llvm::Expected<std::shared_ptr<ast::Expression>> expr = parseExpression();
    if (auto e = expr.takeError()) {
      llvm::errs() << "failed to parse expression in array type : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    arr.setExpression(*expr);

    if (check(TokenKind::SquareClose)) {
      assert(eat(TokenKind::SquareClose));
      return std::make_shared<ArrayType>(arr);
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse array or slice type");
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseInferredType() {
  Location loc = getLocation();

  InferredType infer = {loc};

  if (check(TokenKind::Underscore)) {
    assert(eat(TokenKind::Underscore));
    return std::make_shared<InferredType>(infer);
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse inferred type");
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseNeverType() {
  Location loc = getLocation();

  NeverType never = {loc};

  if (check(TokenKind::Not)) {
    assert(eat(TokenKind::Not));
    return std::make_shared<NeverType>(never);
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse never type");
}

llvm::Expected<ast::GenericArgs> Parser::parseGenericArgs() { assert(false); }

llvm::Expected<ast::types::TypeParamBound> Parser::parseTypeParamBound() {
  Location loc = getLocation();

  // TypeParamBound bound = {};
}

llvm::Expected<ast::types::TypeParamBounds> Parser::parseTypeParamBounds() {
  Location loc = getLocation();

  TypeParamBounds bounds = {loc};

  llvm::Expected<ast::types::TypeParamBound> first = parseTypeParamBound();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse  type param bound in type param bounds : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bounds.addTypeParamBound(*first);

  // FIXME
}

PathKind Parser::testTypePathOrSimplePath() {
  if (check(TokenKind::PathSep)) {
    assert(eat(TokenKind::PathSep));
  }

  while (true) {
    if (check(TokenKind::Identifier)) {
      assert(eat(TokenKind::Identifier));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SUPER)) {
      assert(eatKeyWord(KeyWordKind::KW_SUPER));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SELFVALUE)) {
      assert(eatKeyWord(KeyWordKind::KW_SELFVALUE));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_CRATE)) {
      assert(eatKeyWord(KeyWordKind::KW_CRATE));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_DOLLARCRATE)) {
      assert(eatKeyWord(KeyWordKind::KW_DOLLARCRATE));
      continue;
    } else if (checkKeyWord(KeyWordKind::KW_SELFTYPE)) {
      return PathKind::TypePath;
    } else if (check(TokenKind::Lt)) {
      // GenericArgs
      return PathKind::TypePath;
    } else if (check(TokenKind::ParenOpen)) {
      // TypePathFn
      return PathKind::TypePath;
    } else if (check(TokenKind::PathSep)) {
      assert(eat(TokenKind::PathSep));
      continue;
    } else if (check(TokenKind::Eof)) {
      return PathKind::Unknown;
    } else {
      return PathKind::SimplePath;
    }
  }
  // SimplePath?
}
// TypePath: PathIdentSegment
// SimplePath: SimplePathSegment

// eof

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
    parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType() {
  if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ForLifetimes> forLifetimes = parseForLifetimes();
    // check error

    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      // BareFunctionType
    } else if (check(TokenKind::PathSep)) {
      // TraitObjectType
    }
  } else { // not for
    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      // BareFunctionType
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      // BareFunctionType
    } else if (check(TokenKind::PathSep)) {
      // TraitObjectType: TypePath
      // TypePath: TypePath
      // MacroInvocation: SimplePath
    }
    // BareFunctionType or TraitObjectType
  }
  // else if (check(TokenKind::ParenOpen)) {
  //   // ParenType or TupleType or TraitObjectType
  //   //    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> types
  //   =
  //   //       parseType();
  // }
  // else if (check(TokenKind::PathSeq)) {
  //   // SimplePath in MacroInvocation
  //   // TypePath
  //   // Or TraitObjectType
  // }
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseType() {
  if (check(TokenKind::Star))
    return parseRawPointerType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

  if (checkKeyWord(KeyWordKind::KW_IMPL))
    return parseImplType();

  if (check(TokenKind::Not))
    return parseNeverType();

  if (check(TokenKind::And))
    return parseReferenceType();

  if (check(TokenKind::Underscore))
    return parseInferredType();

  if (check(TokenKind::Lt))
    return parseQualifiedPathInType();

  if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1))
    return parseTupleType();

  if (checkKeyWord(KeyWordKind::KW_UNSAFE))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_EXTERN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_FN))
    return parseBareFunctionType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

  if (check(TokenKind::QMark))
    return parseTraitObjectType();
}

} // namespace rust_compiler::parser

/*
  TypePath or MacroInvocation
  TraitObjectType
  BareFunctionType with ForLifetimes
 */
