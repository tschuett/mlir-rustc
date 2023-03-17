#include "ADT/CanonicalPath.h"
#include "AST/PathIdentSegment.h"
#include "AST/Types/TupleType.h"
#include "AST/Types/TypePath.h"
#include "AST/Types/TypePathSegment.h"
#include "Location.h"
#include "Resolver.h"

#include <memory>

using namespace rust_compiler::sema::type_checking;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::generateBuiltins() {
  // unsigned integer
  u8 = std::make_unique<TyTy::UintType>(mappings->getNextNodeId(),
                                        TyTy::UintKind::U8);
  setupBuiltin("u8", u8.get());
  u16 = std::make_unique<TyTy::UintType>(mappings->getNextNodeId(),
                                         TyTy::UintKind::U16);
  setupBuiltin("u16", u16.get());
  u32 = std::make_unique<TyTy::UintType>(mappings->getNextNodeId(),
                                         TyTy::UintKind::U32);
  setupBuiltin("u32", u32.get());
  u64 = std::make_unique<TyTy::UintType>(mappings->getNextNodeId(),
                                         TyTy::UintKind::U64);
  setupBuiltin("u64", u64.get());
  u128 = std::make_unique<TyTy::UintType>(mappings->getNextNodeId(),
                                          TyTy::UintKind::U128);
  setupBuiltin("u128", u128.get());

  // signed integer
  i8 = std::make_unique<TyTy::IntType>(mappings->getNextNodeId(),
                                       TyTy::IntKind::I8);
  setupBuiltin("i8", i8.get());
  i16 = std::make_unique<TyTy::IntType>(mappings->getNextNodeId(),
                                        TyTy::IntKind::I16);
  setupBuiltin("i16", i16.get());
  i32 = std::make_unique<TyTy::IntType>(mappings->getNextNodeId(),
                                        TyTy::IntKind::I32);
  setupBuiltin("i32", i32.get());
  i64 = std::make_unique<TyTy::IntType>(mappings->getNextNodeId(),
                                        TyTy::IntKind::I64);
  setupBuiltin("i64", i64.get());
  i128 = std::make_unique<TyTy::IntType>(mappings->getNextNodeId(),
                                         TyTy::IntKind::I128);
  setupBuiltin("i128", i128.get());

  // float
  f32 = std::make_unique<TyTy::FloatType>(mappings->getNextNodeId(),
                                          TyTy::FloatKind::F32);
  setupBuiltin("f32", f32.get());
  f64 = std::make_unique<TyTy::FloatType>(mappings->getNextNodeId(),
                                          TyTy::FloatKind::F64);
  setupBuiltin("f64", f64.get());

  // bool
  rbool = std::make_unique<TyTy::BoolType>(mappings->getNextNodeId());
  setupBuiltin("bool", rbool.get());

  // usize and isize
  usize = std::make_unique<TyTy::USizeType>(mappings->getNextNodeId());
  setupBuiltin("usize", usize.get());
  isize = std::make_unique<TyTy::ISizeType>(mappings->getNextNodeId());
  setupBuiltin("isize", isize.get());

  // char and str
  charType = std::make_unique<TyTy::CharType>(mappings->getNextNodeId());
  setupBuiltin("char", charType.get());
  strType = std::make_unique<TyTy::StrType>(mappings->getNextNodeId());
  setupBuiltin("str", strType.get());

  never = std::make_unique<TyTy::NeverType>(mappings->getNextNodeId());
  setupBuiltin("!", never.get());

  TyTy::TupleType *unitType =
      TyTy::TupleType::getUnitType(mappings->getNextNodeId());

  emptyTupleType =
      new ast::types::TupleType(Location::getBuiltinLocation());
  builtins.push_back(emptyTupleType);
  tyctx->insertBuiltin(unitType->getReference(), emptyTupleType->getNodeId(),
                       unitType);
  setUnitTypeNodeId(emptyTupleType->getNodeId());
}

void Resolver::setupBuiltin(std::string_view name, TyTy::BaseType *tyty) {
  PathIdentSegment seg = {Location::getBuiltinLocation()};
  seg.setIdentifier(name);
  types::TypePathSegment typeSeg = {Location::getBuiltinLocation()};
  typeSeg.setSegment(seg);

  TypePath *builtinType = new types::TypePath(Location::getBuiltinLocation());
  builtinType->addSegment(typeSeg);

  builtins.push_back(builtinType);
  tyctx->insertBuiltin(tyty->getReference(), builtinType->getNodeId(), tyty);
  // FIXME
  //mappings->insertNodeToHir(builtinType->getNodeId(), tyty->getReference());
  mappings->insertCanonicalPath(
      builtinType->getNodeId(),
      CanonicalPath::newSegment(builtinType->getNodeId(), name));
}

} // namespace rust_compiler::sema::resolver
