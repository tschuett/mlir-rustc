// marker trait from std::marker::Tuple
pub trait Tuple {}

pub trait FnOnce<Args>
where
    Args: Tuple,
{
    type Output;

    // Required method
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

pub fn const_eval_select<ARG: Tuple, F, G, RET>(
    arg: ARG,
    called_in_const: F,
    called_at_rt: G,
) -> RET
where
    G: FnOnce<ARG, Output = RET>,
    F: FnOnce<ARG, Output = RET>;

pub const fn inconsistent() -> i32 {
    fn runtime() -> i32 {
        1
    }
    const fn compiletime() -> i32 {
        2
    }

    unsafe {
        // and `runtime`.
        const_eval_select((), compiletime, runtime)
    }
}

fn main() {
    inconsistent();
    ()
}
