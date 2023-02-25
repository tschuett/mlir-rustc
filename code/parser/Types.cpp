#include "AST/Types/ArrayType.h"
#include "AST/Types/ImplTraitType.h"
#include "AST/Types/ImplTraitTypeOneBound.h"
#include "AST/Types/InferredType.h"
#include "AST/Types/NeverType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TypeParamBound.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::ast;
using namespace llvm;

namespace rust_compiler::parser {

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
    parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType() {
  Location loc = getLocation();
  CheckPoint cp = getCheckPoint();

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    return parseTraitObjectType();
  } else if (check(TokenKind::ParenOpen) && check(TokenKind::ParenClose, 1)) {
    return parseTupleType();
  } else if (check(TokenKind::ParenOpen)) {
    return parseTupleOrParensType();
  } else if (check(TokenKind::QMark)) {
    return parseTraitObjectType();
  } else if (checkKeyWord(KeyWordKind::KW_FN)) {
    return parseBareFunctionType();
  } else if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
    return parseBareFunctionType();
  } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
    return parseBareFunctionType();
  } else if (check(TokenKind::PathSep)) {

    while (true) {
      if (check(TokenKind::Eof)) {
      } else if (check(TokenKind::PathSep)) {
        assert(eat(TokenKind::PathSep));
      } else if (check(TokenKind::Not)) {
        recover(cp);
        return parseMacroInvocationType();
      } else if (!checkSimplePathSegment()) {
        recover(cp);
        return parsePathType();
      } else {
        recover(cp);
        return parseMacroInvocationType();
      }
    }
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
    if (auto e = forL.takeError()) {
      llvm::errs() << "failed to parse for lifetimes in  : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
      recover(cp);
      return parseBareFunctionType();
    } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
      recover(cp);
      return parseBareFunctionType();
    } else if (checkKeyWord(KeyWordKind::KW_FN)) {
      recover(cp);
      return parseBareFunctionType();
    }
    recover(cp);
    return parseTraitObjectType();
  } else if (checkLifetime()) {
    recover(cp);
    return parseTraitObjectType();
  } else {
    while(true) {
      if (check(TokenKind::Eof)) {
      } else if (check(TokenKind::Plus)) {
        recover(cp);
        return parseTraitObjectType();
      } else if (check(TokenKind::Plus)) {
      }
    }
  }
  // FIXME: probably buggy
}
  /*
    TypePath
    MacroInvocation
    TraitObject
    done: BareFunctionType
   */


llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseImplTraitTypeOneBound() {
  Location loc = getLocation();
  ImplTraitTypeOneBound one = {loc};

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(
        inconvertibleErrorCode(),
        "failed to parse impl keyword in impl trait type one bound");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
      parseTraitBound();
  if (auto e = bound.takeError()) {
    llvm::errs()
        << "failed to parse trait bound in impl trait type one bound : "
        << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  one.setBound(*bound);

  return std::make_shared<ImplTraitTypeOneBound>(one);
}

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

  CheckPoint cp = getCheckPoint();

  if (!checkKeyWord(KeyWordKind::KW_IMPL)) {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse impl keyword in impl trait");
  }
  assert(eatKeyWord(KeyWordKind::KW_IMPL));

  while (true) {
    if (check(TokenKind::LIFETIME_OR_LABEL)) {
      // TypeParamBounds
      recover(cp);
      return parseImplTraitType();
    } else if (check(TokenKind::LIFETIME_TOKEN) &&
               checkStaticOrUnderscore()) { // 'static or '_ FIXME
      // TypeParamBound
      recover(cp);
      return parseImplTraitType();
    } else if (check(TokenKind::QMark) || checkKeyWord(KeyWordKind::KW_FOR) ||
               check(TokenKind::DoubleColon) || checkPathIdentSegment()) {
      // TraitBound
      llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (auto e = bound.takeError()) {
        llvm::errs() << "failed to parse trait bound in impl type : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::Plus)) {
        recover(cp);
        return parseImplTraitType();
      } else {
        recover(cp);
        return parseImplTraitTypeOneBound();
      }
    }
  }
}

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseTraitObjectType() {
  Location loc = getLocation();

  CheckPoint cp = getCheckPoint();

  if (checkKeyWord(KeyWordKind::KW_DYN)) {
    assert(eatKeyWord(KeyWordKind::KW_DYN));
  }

  while (true) {
    if (check(TokenKind::LIFETIME_OR_LABEL)) {
      // TypeParamBounds
      recover(cp);
      return parseTraitObjectType();
    } else if (check(TokenKind::LIFETIME_TOKEN) &&
               checkStaticOrUnderscore()) { // 'static or '_ FIXME
      // TypeParamBound
      recover(cp);
      return parseTraitObjectType();
    } else if (check(TokenKind::QMark) || checkKeyWord(KeyWordKind::KW_FOR) ||
               check(TokenKind::DoubleColon) || checkPathIdentSegment()) {
      // TraitBound
      llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> bound =
          parseTraitBound();
      if (auto e = bound.takeError()) {
        llvm::errs() << "failed to parse trait bound in impl type : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      if (check(TokenKind::Plus)) {
        recover(cp);
        return parseTraitObjectType();
      } else {
        recover(cp);
        return parseTraitObjectTypeOneBound();
      }
    }
  }
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

llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseTypeParamBound() {
  Location loc = getLocation();

  if (check(TokenKind::ParenOpen)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (check(TokenKind::QMark)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (checkKeyWord(KeyWordKind::KW_FOR)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (check(TokenKind::DoubleColon)) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  } else if (checkPathIdentSegment()) {
    llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> trait =
        parseTraitBound();
    if (auto e = trait.takeError()) {
      llvm::errs() << "failed to parse trait bound in type param bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    return *trait;
  }

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> life =
      parseLifetime();
  if (auto e = life.takeError()) {
    llvm::errs() << "failed to parse lifetime in type param bound : "
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  return *life;
}

llvm::Expected<ast::types::TypeParamBounds> Parser::parseTypeParamBounds() {
  Location loc = getLocation();

  TypeParamBounds bounds = {loc};

  llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> first =
      parseTypeParamBound();
  if (auto e = first.takeError()) {
    llvm::errs() << "failed to parse  type param bound in type param bounds :"
                 << toString(std::move(e)) << "\n";
    exit(EXIT_FAILURE);
  }
  bounds.addTypeParamBound(*first);

  while (true) {
    if (check(TokenKind::Eof)) {
      // abort
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse type param bounds: eof");
    } else if (!check(TokenKind::Plus)) {
      assert(eat(TokenKind::Plus));
      if (checkTypeParamBound()) {
        llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>> type =
            parseTypeParamBound();
        if (auto e = first.takeError()) {
          llvm::errs()
              << "failed to parse  type param bound in type param bounds :"
              << toString(std::move(e)) << "\n";
          exit(EXIT_FAILURE);
        }
        bounds.addTypeParamBound(*type);
      } else {
        bounds.setTrailingPlus();
        return bounds;
      }
    } else {
      // done
      return bounds;
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse type param bounds");
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

// llvm::Expected<std::shared_ptr<ast::types::TypeExpression>> Parser::
//     parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType()
//     {
//   CheckPoint qp = getCheckPoint();
//
//   if (checkKeyWord(KeyWordKind::KW_FOR)) {
//     llvm::Expected<ForLifetimes> forLifetimes = parseForLifetimes();
//     if (auto e = forLifetimes.takeError()) {
//       llvm::errs() << "failed to parse for lifetime in ... : "
//                    << toString(std::move(e)) << "\n";
//       exit(EXIT_FAILURE);
//     }
//
//     if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_FN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (check(TokenKind::PathSep)) {
//       // TraitObjectType
//       // TypePath
//       // MacroInvocation
//     } else if (check(TokenKind::)) {
//     }
//   } else { // not for
//     if (checkKeyWord(KeyWordKind::KW_UNSAFE)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_EXTERN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (checkKeyWord(KeyWordKind::KW_FN)) {
//       recover(qp);
//       return parseBareFunctionType();
//     } else if (check(TokenKind::PathSep)) {
//       // TraitObjectType: TypePath
//       // TypePath: TypePath
//       // MacroInvocation: SimplePath
//     } else if () {
//     }
//     // BareFunctionType or TraitObjectType
//   }
//   // else if (check(TokenKind::ParenOpen)) {
//   //   // ParenType or TupleType or TraitObjectType
//   //   //    llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
//   types
//   //   =
//   //   //       parseType();
//   // }
//   // else if (check(TokenKind::PathSeq)) {
//   //   // SimplePath in MacroInvocation
//   //   // TypePath
//   //   // Or TraitObjectType
//   // }
// }

llvm::Expected<std::shared_ptr<ast::types::TypeExpression>>
Parser::parseType() {
  if (checkKeyWord(KeyWordKind::KW_IMPL))
    return parseImplType();

  if (checkKeyWord(KeyWordKind::KW_DYN))
    return parseTraitObjectType();

  if (check(TokenKind::Star))
    return parseRawPointerType();

  if (check(TokenKind::SquareOpen))
    return parseArrayOrSliceType();

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

  return parseTupleOrParensTypeOrTypePathOrMacroInvocationOrTraitObjectTypeOrBareFunctionType();
}

llvm::Expected<std::shared_ptr<ast::types::TypeParamBound>>
Parser::parseTraitBound() {
  Location loc = getLocation();
  TraitBound tr = {loc};

  if (check(TokenKind::ParenOpen)) {
    tr.setHasParenthesis();
    assert(eat(TokenKind::ParenOpen));
    if (check(TokenKind::QMark)) {
      assert(eat(TokenKind::QMark));
      tr.setHasQuestionMark();
    }
    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (auto e = forL.takeError()) {
        llvm::errs() << "failed to parse ForLifetimes in trait bound : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(*forL);
    }
    llvm::Expected<std::shared_ptr<ast::types::TypePath>> path =
        parseTypePath();
    if (auto e = path.takeError()) {
      llvm::errs() << "failed to parse type path in trait bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(*path);
    if (!check(TokenKind::ParenClose))
      return createStringError(inconvertibleErrorCode(),
                               "failed to parse trait bound ;missing ): ");
    assert(eat(TokenKind::ParenClose));
    return std::make_shared<TraitBound>(tr);
  } else if (!check(TokenKind::ParenOpen)) {
    if (check(TokenKind::QMark)) {
      assert(eat(TokenKind::QMark));
      tr.setHasQuestionMark();
    }

    if (checkKeyWord(KeyWordKind::KW_FOR)) {
      llvm::Expected<ast::types::ForLifetimes> forL = parseForLifetimes();
      if (auto e = forL.takeError()) {
        llvm::errs() << "failed to parse ForLifetimes in trait bound : "
                     << toString(std::move(e)) << "\n";
        exit(EXIT_FAILURE);
      }
      tr.setForLifetimes(*forL);
    }

    llvm::Expected<std::shared_ptr<ast::types::TypePath>> path =
        parseTypePath();
    if (auto e = path.takeError()) {
      llvm::errs() << "failed to parse type path in trait bound : "
                   << toString(std::move(e)) << "\n";
      exit(EXIT_FAILURE);
    }
    tr.setTypePath(*path);
    return std::make_shared<TraitBound>(tr);
  } else {
    return createStringError(inconvertibleErrorCode(),
                             "failed to parse trait bound: ");
  }

  return createStringError(inconvertibleErrorCode(),
                           "failed to parse trait bound: ");
}

} // namespace rust_compiler::parser
