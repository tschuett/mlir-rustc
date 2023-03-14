fn main() {
    match s {
        Point { x: 10, y: 20 } => (),
        Point { y: 10, x: 20 } => (), // order doesn't matter
        Point { x: 10, .. } => (),
        Point { .. } => (),
    }

    match t {
        PointTuple { 0: 10, 1: 20 } => (),
        PointTuple { 1: 10, 0: 20 } => (), // order doesn't matter
        PointTuple { 0: 10, .. } => (),
        PointTuple { .. } => (),
    }
}
