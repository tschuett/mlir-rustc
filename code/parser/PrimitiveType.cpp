#include "PrimitiveType.h"

#include "AST/Types/PrimitiveTypes.h"
#include "Lexer/Token.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
tryParsePrimitiveType(std::span<lexer::Token> tokens) {

  std::span<lexer::Token> view = tokens;
  Location loc = tokens.front().getLocation();

  if (view.front().getKind() == lexer::TokenKind::Not)
    return std::make_shared<ast::types::PrimitiveType>(
        ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::Never));

  if (view.front().getKind() == lexer::TokenKind::Integer) {
    switch (view.front().getIntegerKind()) {
    case lexer::IntegerKind::I8: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::I8));
    }
    case lexer::IntegerKind::I16: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::I16));
    }
    case lexer::IntegerKind::I32: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::I32));
    }
    case lexer::IntegerKind::I64: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::I64));
    }
    case lexer::IntegerKind::I128: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::I128));
    }
    case lexer::IntegerKind::U8: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::U8));
    }
    case lexer::IntegerKind::U16: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::U16));
    }
    case lexer::IntegerKind::U32: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::U32));
    }
    case lexer::IntegerKind::U64: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::U64));
    }
    case lexer::IntegerKind::U128: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::U128));
    }
    case lexer::IntegerKind::ISize: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::Isize));
    }
    case lexer::IntegerKind::USize: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::Usize));
    }
    }
  }

  if (view.front().getKind() == lexer::TokenKind::Float) {
    switch (view.front().getFloatKind()) {
    case lexer::FloatKind::F32: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::F32));
    }
    case lexer::FloatKind::F64: {
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::F64));
    }
    }
  }

  if (view.front().getKind() == lexer::TokenKind::Identifier) {

    if (view.front().getIdentifier() == "bool")
      return std::make_shared<ast::types::PrimitiveType>(
          ast::types::PrimitiveType(loc, types::PrimitiveTypeKind::Boolean));
  }

  // TODO

  return std::nullopt;
}

} // namespace rust_compiler::parser
