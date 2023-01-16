#pragma once

#include "TargetInfo/Struct.h"
#include "TargetInfo/TargetInfo.h"
#include "TargetInfo/Types.h"

#include <span>
#include <vector>

namespace rust_compiler::target_info {

class TargetInfo;

class TypeBuilder {

public:
  virtual ~TypeBuilder() = default;

  virtual ArrayType *getArray(Type *type, size_t elements);

  virtual BitfieldType *getBitfield(Type *type, size_t bits);

  virtual BuiltinType *getBuiltin(BuiltinKind bk);

  virtual FunctionType *getFunction(Signature sig);

  virtual PointerType *getPointer(BuiltinKind baseType);

  virtual ScalableVectorType *getScalableVector(BuiltinKind baseType);

  virtual ScalablePredicateType *getScalablePredicate(BuiltinKind baseType);

  virtual StructType *getStruct(std::span<Type *> members) = 0;

  virtual UnionType *getUnion(std::span<Type *> members);

  virtual VectorType *getVector(size_t bits);

protected:
  TargetInfo *target;

  TypeBuilder(TargetInfo *target) : target(target) {}

  virtual bool areScalableTypesSupported() const = 0;

  virtual bool isSupportedVectorLength(size_t bits) const = 0;

  virtual bool isSupportedBitFieldType(const Type *) const = 0;
};

} // namespace rust_compiler::target_info
