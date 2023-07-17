#pragma once

#include "AST/Types/RawPointerType.h"
#include "AST/Types/ReferenceType.h"
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

  std::string mangleType(const ast::types::TypeExpression *);
  std::string mangleTypePath(const ast::types::TypePath *);
  std::string mangleConst(uint64_t);
  std::string mangleBareFunctionType(const ast::types::BareFunctionType *);

  std::optional<std::string> tryBasicType(const ast::types::TypePathSegment &);
  std::string mangleTupleType(const ast::types::TupleType *);
  std::string mangleRawPointerType(const ast::types::RawPointerType *);
  std::string mangleReferenceType(const ast::types::ReferenceType *);
  std::string mangleLifetime(const ast::Lifetime& );
  std::string mangleBackref();
};

} // namespace rust_compiler::mangler
