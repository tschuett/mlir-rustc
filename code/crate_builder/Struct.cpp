#include "AST/AssociatedItem.h"
#include "AST/Implementation.h"
#include "AST/TupleStruct.h"
#include "CrateBuilder/CrateBuilder.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ExtensibleDialect.h>
#include <mlir/Support/LogicalResult.h>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

void CrateBuilder::emitStruct(ast::Struct *struc) {
  switch (struc->getKind()) {
  case StructKind::StructStruct2: {
    emitStructStruct(static_cast<StructStruct *>(struc));
    break;
  }
  case StructKind::TupleStruct2: {
    emitTupleStruct(static_cast<TupleStruct *>(struc));
    break;
  }
  }
}

void CrateBuilder::emitStructStruct(ast::StructStruct *stru) {
  assert(false);

  mlir::ExtensibleDialect *hirDialect =
      builder.getContext()->getOrLoadDialect<rust_compiler::hir::HirDialect>();

  auto verifier =
      [](llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
         llvm::ArrayRef<mlir::Attribute> args) -> mlir::LogicalResult {
    return mlir::LogicalResult::success();
  };

  std::string name = stru->getIdentifier().toString();
  std::unique_ptr<mlir::DynamicTypeDefinition> aStruct =
      mlir::DynamicTypeDefinition::get(name, std::move(hirDialect),
                                       std::move(verifier));

  hirDialect->registerDynamicType(std::move(aStruct));

  mlir::MemRefType::Builder builder = {{1},
                                       mlir::DynamicType::get(aStruct.get())};
  mlir::MemRefType memRef = builder;

  StructType structType = {stru, memRef, mlir::DynamicType::get(aStruct.get()),
                           name};
}

void CrateBuilder::emitTupleStruct(ast::TupleStruct *) { assert(false); }

void CrateBuilder::emitImplementation(ast::Implementation *impl) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    emitInherentImpl(static_cast<InherentImpl *>(impl));
    break;
  }
  case ImplementationKind::TraitImpl: {
    emitTraitImpl(static_cast<TraitImpl *>(impl));
    break;
  }
  }
}

void CrateBuilder::emitInherentImpl(ast::InherentImpl *impl) {
  for (auto &asso : impl->getAssociatedItems())
    emitInherentAssoItem(impl, asso);
}

void CrateBuilder::emitTraitImpl(ast::TraitImpl *) { assert(false); }

void CrateBuilder::emitInherentAssoItem(InherentImpl *impl,
                                        ast::AssociatedItem &asso) {
  switch (asso.getKind()) {
  case AssociatedItemKind::ConstantItem: {
    break;
  }
  case AssociatedItemKind::Function: {
    Function *fun = static_cast<Function *>(
        static_cast<VisItem *>(asso.getFunction().get()));
    if (fun->isMethod()) {
      emitInherentMethod(fun, impl);
    } else {
      assert(false);
    }
    break;
  }
  case AssociatedItemKind::MacroInvocationSemi: {
    break;
  }
  case AssociatedItemKind::TypeAlias: {
    break;
  }
  }
  assert(false);
}

void CrateBuilder::emitInherentMethod(ast::Function *, ast::InherentImpl *) {}

mlir::FunctionType CrateBuilder::getMethodType(ast::Function *fun,
                                               mlir::MemRefType memRef) {

  llvm::SmallVector<mlir::Type> parameterType;
  parameterType.push_back(memRef);
  FunctionParameters params = fun->getParams();
  std::vector<FunctionParam> parms = params.getParams();
  for (FunctionParam &parm : parms) {
    switch (parm.getKind()) {
    case FunctionParamKind::Pattern: {
      FunctionParamPattern pattern = parm.getPattern();
      assert(pattern.hasType());
      parameterType.push_back(getType(pattern.getType().get()));
      break;
    }
    case FunctionParamKind::DotDotDot: {
      assert(false && "to be implemented later");
    }
    case FunctionParamKind::Type: {
      assert(false && "to be implemented later");
    }
    }
  }

  mlir::Type returnType = builder.getNoneType();

  if (fun->hasReturnType())
    returnType = getType(fun->getReturnType().get());

  mlir::TypeRange inputs = parameterType;
  mlir::TypeRange results = {returnType};

  return builder.getFunctionType(inputs, results);
}

} // namespace rust_compiler::crate_builder
