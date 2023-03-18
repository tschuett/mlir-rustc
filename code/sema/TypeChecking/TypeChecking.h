#pragma once

#include "AST/Crate.h"
#include "AST/Item.h"
#include "AST/MacroItem.h"
#include "AST/Types/TypeExpression.h"
#include "AST/WhereClause.h"
#include "Basic/Ids.h"
#include "TyCtx/TyCtx.h"
#include "TyTy.h"

#include <map>
#include <memory>
#include <vector>

namespace rust_compiler::sema::type_checking {

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html
class TypeResolver {
public:
  TypeResolver();

  void checkCrate(std::shared_ptr<ast::Crate> crate);

private:
  void checkVisItem(std::shared_ptr<ast::VisItem> v);
  void checkMacroItem(std::shared_ptr<ast::MacroItem> v);
  void checkFunction(std::shared_ptr<ast::Function> f);
  std::optional<TyTy::BaseType *>
      checkType(std::shared_ptr<ast::types::TypeExpression>);
  void checkWhereClause(const ast::WhereClause &);
  void checkExpression(std::shared_ptr<ast::Expression>);

  tyctx::TyCtx *tcx;
};

} // namespace rust_compiler::sema::type_checking
