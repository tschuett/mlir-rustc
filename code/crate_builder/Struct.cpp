#include "AST/AssociatedItem.h"
#include "AST/Implementation.h"
#include "AST/TupleStruct.h"
#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"
#include "CrateBuilder/StructType.h"
#include "Hir/HirTypes.h"
#include "Mangler/Mangler.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ExtensibleDialect.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>

using namespace rust_compiler::ast;
using namespace llvm;

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

  //  mlir::Dialect *hirDialect =
  //      builder.getContext()->getOrLoadDialect<rust_compiler::hir::HirDialect>();

  mangler::Mangler mangler = {crate};
  std::string mangledName = mangler.mangleStruct(visItemStack, crate);

  SmallVector<mlir::Type> members;

  StructFields fields = stru->getFields();

  for (const StructField &field : fields.getFields())
    members.push_back(getType(field.getType().get()));

  mlir::Type structType = rust_compiler::hir::StructType::get(
      builder.getContext(), members, mangledName, false);
  //  mlir::Type structType = builder.create<hir::StructType>(
  //      getLocation(stru->getLocation()), members, mangledName, false);

  mlir::MemRefType memRefType = mlir::MemRefType::Builder({1}, structType);
  structTypes[stru->getNodeId()] = {structType, memRefType};
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
  std::optional<basic::NodeId> typeId =
      tyCtx->lookupResolvedType(impl->getType()->getNodeId());
  if (!typeId) {
    llvm::errs() << "emitInherentImpl could not find type" << '\n';
    exit(EXIT_FAILURE);
  }

  if (!structTypes.contains(*typeId)) {
    llvm::errs() << "emitInherentImpl could not find struct" << '\n';
    exit(EXIT_FAILURE);
  }

  for (auto &asso : impl->getAssociatedItems())
    emitInherentAssoItem(impl, asso, structTypes[*typeId].second);
}

void CrateBuilder::emitTraitImpl(ast::TraitImpl *) { assert(false); }

void CrateBuilder::emitInherentAssoItem(InherentImpl *impl,
                                        ast::AssociatedItem &asso,
                                        mlir::MemRefType memRefStructType) {
  switch (asso.getKind()) {
  case AssociatedItemKind::ConstantItem: {
    break;
  }
  case AssociatedItemKind::Function: {
    ast::Function *fun = static_cast<ast::Function *>(
        static_cast<VisItem *>(asso.getFunction().get()));
    if (fun->isMethod()) {
      emitInherentMethod(fun, impl, memRefStructType);
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

void CrateBuilder::emitInherentMethod(ast::Function *f, ast::InherentImpl *,
                                      mlir::MemRefType memRef) {
  assert(false);

  llvm::ScopedHashTableScope<basic::NodeId, mlir::Value> scope(symbolTable);
  llvm::ScopedHashTableScope<basic::NodeId, mlir::Value> allocaScope(
      allocaTable);

  builder.setInsertionPointToEnd(theModule.getBody());

  mlir::FunctionType funType = getMethodType(f, memRef);

  // FIXME add mangled name
  mlir::func::FuncOp fun = builder.create<mlir::func::FuncOp>(
      getLocation(f->getLocation()), f->getName().toString(), funType);
  assert(fun != nullptr);

  llvm::SmallVector<basic::NodeId> parameterNames;
  llvm::SmallVector<mlir::Type> parameterTypes;
  llvm::SmallVector<mlir::Location> parameterLocations;
  // FIXME xxx memref
  for (FunctionParam &parm : f->getParams().getParams()) {
    if (parm.getKind() == FunctionParamKind::Pattern) {
      parameterNames.push_back(parm.getPattern().getPattern()->getNodeId());
      parameterTypes.push_back(getType(parm.getPattern().getType().get()));
      parameterLocations.push_back(getLocation(parm.getLocation()));
    } else {
      assert(false);
    }
  }

  mlir::Block *entryBlock = builder.createBlock(
      &fun.getBody(), {}, parameterTypes, parameterLocations);

  for (const auto namedValue :
       llvm::zip(parameterNames, entryBlock->getArguments()))
    declare(std::get<0>(namedValue), std::get<1>(namedValue));
}

mlir::FunctionType CrateBuilder::getMethodType(ast::Function *fun,
                                               mlir::MemRefType memRef) {

  // fixme memref
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
