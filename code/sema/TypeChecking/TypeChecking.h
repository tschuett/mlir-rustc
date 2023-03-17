#pragma once

#include "AST/Crate.h"
#include "Basic/Ids.h"
#include "TyTy.h"

#include <memory>
#include <map>
#include <vector>

namespace rust_compiler::sema::type_checking {

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html
class TypeCheckContext {
public:
  static TypeCheckContext *get();

  void checkCrate(std::shared_ptr<ast::Crate>);

  void insertBuiltin(basic::NodeId nodeId, basic::NodeId reference, TyTy::BaseType *type);

private:
  std::map<basic::NodeId, basic::NodeId> nodeToTypeReference;
  std::map<basic::NodeId, TyTy::BaseType *> resolvedTypes;
  std::vector<std::unique_ptr<TyTy::BaseType>> builtinTypes;
};

} // namespace rust_compiler::sema::type_checking
