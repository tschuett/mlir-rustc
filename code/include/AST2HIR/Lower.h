#pragma once

#include "AST/Crate.h"
#include "Hir/Crate.h"

#include <memory>

namespace rust_compiler::ast2hir {

class ASTLowering {
public:
  static std::unique_ptr<HIR::Crate> Resolve(AST::Crate &astCrate);
  ~ASTLowering();

private:
  ASTLowering(AST::Crate &astCrate);
  std::unique_ptr<HIR::Crate> go();

  AST::Crate &astCrate;
};

} // namespace rust_compiler::ast2hir
