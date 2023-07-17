#pragma once

#include "AST/GenericArgsConst.h"
#include "AST/PathIdentSegment.h"
#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/TypePathFn.h"
#include "ConstantEvaluation/ConstantEvaluation.h"

#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <span>
#include <string>

namespace rust_compiler::ast {
class VisItem;
class Crate;
class ClosureExpression;

namespace types {
class TypeExpression;
class TypePath;
class BareFunctionType;
class TypePathSegment;
} // namespace types

} // namespace rust_compiler::ast

namespace rust_compiler::mangler {

/// https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html
/// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Demangle/RustDemangle.cpp
class Mangler {
  constant_evaluation::ConstantEvaluation evaluator;

public:
  Mangler(const ast::Crate *crate) : evaluator(crate){};

  std::string mangleFreestandingFunction(std::span<const ast::VisItem *> path,
                                         ast::Crate *crate);
  std::string mangleStatic(std::span<const ast::VisItem *> path,
                           ast::Crate *crate);

  std::string mangleMethod(std::span<const ast::VisItem *> path,
                           ast::Crate *crate);

  std::string mangleClosure(std::span<const ast::VisItem *> path,
                            ast::Crate *crate, ast::ClosureExpression *closure);

private:
  std::string mangle(std::span<const ast::VisItem *> path, ast::Crate *crate);

  /// not const because of ConstantEvaluation is not const
  std::string mangleType(const ast::types::TypeExpression *) ;
  std::string mangleTypePath(const ast::types::TypePath *);
  std::string mangleConst(uint64_t) const;
  std::string mangleBareFunctionType(const ast::types::BareFunctionType *);

  std::optional<std::string> tryBasicType(const ast::types::TypePathSegment &) const;
  std::string mangleTupleType(const ast::types::TupleType *);
  std::string mangleRawPointerType(const ast::types::RawPointerType *);
  std::string mangleReferenceType(const ast::types::ReferenceType *) ;
  std::string mangleLifetime(const ast::Lifetime &) const;
  std::string mangleBackref() const;
  std::string manglePathIdentSegment(const ast::PathIdentSegment &) const;
  std::string mangleGenericArgs(const ast::GenericArgs &);
  std::string mangleTypePathFunction(const ast::types::TypePathFn &) const;
  std::string mangleGenericArg(const ast::GenericArg &);
  std::string mangleGenericArgsConst(const ast::GenericArgsConst &) const;
};

} // namespace rust_compiler::mangler
