use dezero::*;
use ndarray::prelude::*;

fn main() {
    test_add_twice();
}

pub fn test_forward_prop() {
    let x = Variable::new(arr0(0.5));
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    println!("y = {:?}", y.get_data().view().into_scalar());
}

pub fn test_back_prop() {
    let x = Variable::new(arr0(0.5));
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap().view().into_scalar());
}

pub fn test_numerical_diff() {
    let f = square;
    let x = Variable::new(arr0(2.0));
    let dy = numerical_diff(f, x, None);
    println!("dy = {:?}", dy.view().into_scalar());
}

pub fn test_add() {
    let x = Variable::new(arr0(2.0));
    let y = Variable::new(arr0(3.0));
    let z = add(&square(&x), &square(&y));

    z.backward();
    println!("z = {:?}", z.get_data().view().into_scalar());
    println!("x.grad = {:?}", x.get_grad().unwrap().view().into_scalar());
    println!("y.grad = {:?}", y.get_grad().unwrap().view().into_scalar());
}

pub fn test_add_twice() {
    let x = Variable::new(arr0(3.0));
    let y = add(&x, &x);
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap().view().into_scalar()); // 2.0

    x.clear_grad();
    let y = add(&x, &add(&x, &x));
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap().view().into_scalar()); // 3.0
}

#[cfg(test)]
mod square_test {
    use dezero::*;
    use ndarray::prelude::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_forward() {
        let x = Variable::new(arr0(2.0));
        let y = square(&x);
        let expected = arr0(4.0);
        assert_eq!(y.get_data(), expected);
    }

    #[test]
    fn test_backward() {
        let x = Variable::new(arr0(3.0));
        let y = square(&x);
        y.backward();
        let expected = arr0(6.0);
        assert_eq!(x.get_grad().unwrap(), expected);
    }

    #[test]
    fn test_gradient_check() {
        let arr = Array::random((), Uniform::new(0., 1.));
        let x = Variable::new(arr.clone());
        let y = square(&x);
        y.backward();
        let num_grad = numerical_diff(square, Variable::new(arr), None);
        assert!(x.get_grad().unwrap().abs_diff_eq(&num_grad, 1e-6));
    }
}
