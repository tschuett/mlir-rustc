#pragma once

#include "AST/ClosureExpression.h"
#include "AST/Implementation.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypePath.h"

#include <llvm/Support/raw_ostream.h>
#include <span>
#include <string>

namespace rust_compiler::ast {
class VisItem;
class Crate;
} // namespace rust_compiler::ast

namespace rust_compiler::mangler {

/// https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html
class Mangler {
public:
  Mangler() = default;

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

  std::string mangleType(ast::types::TypeExpression *);
  std::string mangleTypePath(ast::types::TypePath *);
};

} // namespace rust_compiler::mangler
