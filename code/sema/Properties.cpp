#include "Sema/Properties.h"

namespace rust_compiler::sema {

std::string Property2String(PropertyKind kind) {
  switch (kind) {
  case PropertyKind::ADD:
    return "add";
  case PropertyKind::SUBTRACT:
    return "sub";
  case PropertyKind::MULTIPLY:
    return "mul";
  case PropertyKind::DIVIDE:
    return "div";
  case PropertyKind::REMAINDER:
    return "rem";
  case PropertyKind::BITAND:
    return "bitand";
  case PropertyKind::BITOR:
    return "bitor";
  case PropertyKind::BITXOR:
    return "bitxor";
  case PropertyKind::SHL:
    return "shl";
  case PropertyKind::SHR:
    return "shr";

  case PropertyKind::NEGATION:
    return "neg";
  case PropertyKind::NOT:
    return "not";

  case PropertyKind::ADD_ASSIGN:
    return "add_assign";
  case PropertyKind::SUB_ASSIGN:
    return "sub_assign";
  case PropertyKind::MUL_ASSIGN:
    return "mul_assign";
  case PropertyKind::DIV_ASSIGN:
    return "div_assign";
  case PropertyKind::REM_ASSIGN:
    return "rem_assign";
  case PropertyKind::BITAND_ASSIGN:
    return "bitand_assign";
  case PropertyKind::BITOR_ASSIGN:
    return "bitor_assign";
  case PropertyKind::BITXOR_ASSIGN:
    return "bitxor_assign";
  case PropertyKind::SHL_ASSIGN:
    return "shl_assign";
  case PropertyKind::SHR_ASSIGN:
    return "shr_assign";

  case PropertyKind::DEREF:
    return "deref";
  case PropertyKind::DEREF_MUT:
    return "deref_mut";

  case PropertyKind::INDEX:
    return "index";
  case PropertyKind::INDEX_MUT:
    return "index_mut";

  case PropertyKind::RANGE_FULL:
    return "RangeFull";
  case PropertyKind::RANGE:
    return "Range";
  case PropertyKind::RANGE_TO:
    return "RangeTo";
  case PropertyKind::RANGE_FROM:
    return "RangeFrom";
  case PropertyKind::RANGE_INCLUSIVE:
    return "RangeInclusive";
  case PropertyKind::RANGE_TO_INCLUSIVE:
    return "RangeToInclusive";

  case PropertyKind::PHANTOM_DATA:
    return "PhantomData";

  case PropertyKind::FN:
    return "Fn";
  case PropertyKind::FN_MUT:
    return "FnMut";
  case PropertyKind::FN_ONCE:
    return "FnOnce";
  case PropertyKind::FN_ONCE_OUTPUT:
    return "FnOnceOutput";

  case PropertyKind::COPY:
    return "Copy";
  case PropertyKind::CLONE:
    return "Clone";
  case PropertyKind::SIZED:
    return "Sized";

  case PropertyKind::SLICE_ALLOC:
    return "SlizeAlloc";
  case PropertyKind::SLICE_U8_ALLOC:
    return "SlizeU8Alloc";
  case PropertyKind::STR_ALLOC:
    return "StrAlloc";
  case PropertyKind::ARRAY:
    return "Array";
  case PropertyKind::BOOL:
    return "Bool";
  case PropertyKind::CHAR:
    return "Char";
  case PropertyKind::F32:
    return "F32";
  case PropertyKind::F64:
    return "F64";
  case PropertyKind::I8:
    return "I8";
  case PropertyKind::I16:
    return "I16";
  case PropertyKind::I32:
    return "I32";
  case PropertyKind::I64:
    return "I64";
  case PropertyKind::I128:
    return "I128";
  case PropertyKind::ISIZE:
    return "ISIZE";
  case PropertyKind::U8:
    return "U8";
  case PropertyKind::U16:
    return "U16";
  case PropertyKind::U32:
    return "U32";
  case PropertyKind::U64:
    return "U64";
  case PropertyKind::U128:
    return "U128";
  case PropertyKind::USIZE:
    return "USIZE";
  case PropertyKind::CONST_PTR:
    return "ConstPtr";
  case PropertyKind::CONST_SLICE_PTR:
    return "ConstSlicePtr";
  case PropertyKind::MUT_PTR:
    return "MutPtr";
  case PropertyKind::MUT_SLICE_PTR:
    return "MutSlicePtr";
  case PropertyKind::SLICE_U8:
    return "SliceU8";
  case PropertyKind::SLICE:
    return "Slice";
  case PropertyKind::STR:
    return "str";
  case PropertyKind::F32_RUNTIME:
    return "f32Runtime";
  case PropertyKind::F64_RUNTIME:
    return "f64Runtime";

  case PropertyKind::UNKNOWN:
    return "unknown";
  }
}

} // namespace rust_compiler::sema
