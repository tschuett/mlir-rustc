#pragma once

#include "AST/Function.h"
#include "Remarks/OptimizationRemark.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/Remarks/RemarkSerializer.h>
#include <memory>

namespace rust_compiler::remarks {

class OptimizationRemarkEmitter {
public:
  OptimizationRemarkEmitter(const std::shared_ptr<ast::Function> fun,
                            llvm::remarks::RemarkSerializer *serializer)
      : fun(fun), serializer(serializer) {}

  /// Output the remark via the diagnostic handler and to the
  /// optimization record file.
  void emit(OptimizationRemarkBase &OptDiag);

  /// Take a lambda that returns a remark which will be emitted.  Second
  /// argument is only used to restrict this to functions.
  template <typename T>
  void emit(T RemarkBuilder, decltype(RemarkBuilder()) * = nullptr) {
    // Avoid building the remark unless we know there are at least *some*
    // remarks enabled. We can't currently check whether remarks are requested
    // for the calling pass since that requires actually building the remark.

    auto R = RemarkBuilder();
    static_assert(std::is_base_of<OptimizationRemarkBase, decltype(R)>::value,
                  "the lambda passed to emit() must return a remark");
    emit((OptimizationRemarkBase &)R);
  }

private:
  std::shared_ptr<ast::Function> fun;
  llvm::remarks::RemarkSerializer *serializer;
};

} // namespace rust_compiler::remarks

/*
   R.getORE()->emit([&]() {
        std::string type_str;
        llvm::raw_string_ostream rso(type_str);
        Ty->print(rso);
        return OptimizationRemarkMissed(SV_NAME, "UnsupportedType", I0)
               << "Cannot SLP vectorize list: type "
               << rso.str() + " is unsupported by vectorizer";
               });


                 R.getORE()->emit(OptimizationRemark(SV_NAME,
   "StoresVectorized", cast<StoreInst>(Chain[0]))
                     << "Stores SLP vectorized with cost " << NV("Cost", Cost)
                     << " and with tree size "
                     << NV("TreeSize", R.getTreeSize()));
*/
