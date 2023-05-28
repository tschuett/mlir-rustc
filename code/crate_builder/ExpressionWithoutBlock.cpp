#include "AST/ArrayElements.h"
#include "AST/AwaitExpression.h"
#include "AST/CallExpression.h"
#include "AST/Expression.h"
#include "AST/LiteralExpression.h"
#include "AST/PathExpression.h"
#include "AST/ReturnExpression.h"
#include "CrateBuilder/CrateBuilder.h"
#include "Hir/HirOps.h"

#include <cassert>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

namespace rust_compiler::crate_builder {

std::optional<mlir::Value>
CrateBuilder::emitExpressionWithoutBlock(ast::ExpressionWithoutBlock *expr) {
  switch (expr->getWithoutBlockKind()) {
  case ast::ExpressionWithoutBlockKind::LiteralExpression: {
    return emitLiteralExpression(static_cast<LiteralExpression *>(expr));
  }
  case ast::ExpressionWithoutBlockKind::PathExpression: {
    return emitPathExpression(static_cast<PathExpression *>(expr));
  }
  case ast::ExpressionWithoutBlockKind::OperatorExpression: {
    return emitOperatorExpression(static_cast<OperatorExpression *>(expr));
  }
  case ast::ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ArrayExpression: {
    return emitArrayExpression(static_cast<ArrayExpression *>(expr));
  }
  case ast::ExpressionWithoutBlockKind::AwaitExpression: {
    //    std::shared_ptr<ast::AwaitExpression> await =
    //        std::static_pointer_cast<ast::AwaitExpression>(withOut);
    //    mlir::Value body = emitExpression(await->getBody());
    //    return builder.create<hir::AwaitOp>(getLocation(await->getLocation()),
    //                                        body);
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::IndexExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::CallExpression: {
    ast::CallExpression *call = static_cast<ast::CallExpression *>(expr);
    return emitCallExpression(call);
  }
  case ast::ExpressionWithoutBlockKind::MethodCallExpression: {
    ast::MethodCallExpression *call =
        static_cast<ast::MethodCallExpression *>(expr);
    return emitMethodCallExpression(call);
  }
  case ast::ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ReturnExpression: {
    emitReturnExpression(static_cast<ReturnExpression *>(expr));
    return std::nullopt;
  }
  case ast::ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be implemented");
    break;
  }
  }
  assert(false);
}

mlir::Value CrateBuilder::emitLiteralExpression(ast::LiteralExpression *lit) {
  std::optional<tyctx::TyTy::BaseType *> maybeType =
      tyCtx->lookupType(lit->getNodeId());

  switch (lit->getLiteralKind()) {
  case LiteralExpressionKind::CharLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::StringLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::RawStringLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::ByteLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::ByteStringLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::RawByteStringLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::IntegerLiteral: {
    if (maybeType) {

      llvm::APInt result;
      std::string storage = lit->getValue();
      llvm::StringRef(storage).getAsInteger(10, result);

      uint64_t value = result.getLimitedValue();
      mlir::Type type = convertTyTyToMLIR(*maybeType);
      llvm::errs() << (*maybeType)->toString() << "\n";
      llvm::errs() << value << "\n";
      // mlir::IntegerAttr in = mlir::IntegerAttr(type, value);
      // mlir::IntegerAttr::
      mlir::IntegerAttr attr = mlir::IntegerAttr::get(type, value);

      return builder.create<mlir::arith::ConstantOp>(
          getLocation(lit->getLocation()), attr);
    }
    assert(false);
  }
  case LiteralExpressionKind::FloatLiteral: {
    assert(false);
  }
  case LiteralExpressionKind::True: {
    assert(false);
  }
  case LiteralExpressionKind::False: {
    assert(false);
  }
  }
  assert(false);
}

mlir::Value CrateBuilder::emitArrayExpression(ast::ArrayExpression *array) {
  std::optional<tyctx::TyTy::BaseType *> type =
      tyCtx->lookupType(array->getNodeId());
  assert(type.has_value());
  assert((*type)->getKind() == TyTy::TypeKind::Array);

  TyTy::ArrayType *arrayType = static_cast<TyTy::ArrayType *>(*type);

  mlir::Type elementType = convertTyTyToMLIR(arrayType->getElementType());

  if (array->hasArrayElements()) {
    ArrayElements els = array->getArrayElements();

    switch (els.getKind()) {
    case ArrayElementsKind::Repeated: {

      uint64_t count = foldAsUsizeExpression(els.getCount().get());
      std::vector<int64_t> shape = {static_cast<int64_t>(count)};
      std::optional<mlir::Value> value = emitExpression(els.getValue().get());
      assert(value.has_value());
      return builder.create<mlir::vector::BroadcastOp>(
          getLocation(array->getLocation()),
          mlir::VectorType::get(shape, elementType, 0), *value);
    }
    case ArrayElementsKind::List: {
      std::vector<mlir::Value> values;
      for (auto &val : els.getElements()) {
        std::optional<mlir::Value> el = emitExpression(val.get());
        assert(el.has_value());
        values.push_back(*el);
      }
      std::vector<int64_t> shape = {static_cast<int64_t>(values.size())};
      mlir::Type vectorType = mlir::VectorType::get(shape, elementType, 0);

      mlir::Value initialValue = builder.create<mlir::vector::BroadcastOp>(
          getLocation(array->getLocation()), vectorType, values[0]);

      mlir::Value vector = initialValue;
      int idx = 0;
      for (auto &vl : values) {
        mlir::Value index = builder.create<mlir::arith::ConstantOp>(
            getLocation(array->getLocation()),
            builder.getIntegerAttr(builder.getIntegerType(32, false), idx));
        vector = builder.create<mlir::vector::InsertElementOp>(
            getLocation(array->getLocation()), vl, vector, index);
        ++idx;
      }
      return vector;
    }
    }
  }
  assert(false);

  // FIXME: always homogenous array
}

} // namespace rust_compiler::crate_builder
