type i32;
type i64;

struct Point {
    x: i32,
    y: i32,
}

struct TuplePoint(i64, i64);

struct NothingInMe {}

struct Cookie {}

fn main() {
    let p = Point { x: 10, y: 11 };
    let px: i32 = p.x;
    Point { x: 10, y: 20 };
    NothingInMe {};
    TuplePoint(10, 20);
    TuplePoint { 0: 10, 1: 20 }; // Results in the same value as the above line
    const COOKIE: Cookie = Cookie {};
    let c = [Cookie, Cookie {}, Cookie, Cookie {}];
}
