use dezero::*;
use ndarray::prelude::*;

fn main() {
    test_numerical_diff();
    test_forward_prop();
    test_back_prop();
}

fn test_forward_prop() {
    let x = Variable::new(arr0(0.5));
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    println!("y = {:?}", y.get_data());
}

fn test_back_prop() {
    let x = Variable::new(arr0(0.5));
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap());
}

fn test_numerical_diff() {
    let mut f = Square::new();
    let x = Variable::new(arr0(2.0));
    let dy = numerical_diff(&mut f, x, None);
    println!("dy = {:?}", dy);
}
