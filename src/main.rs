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
    println!("y = {:?}", y.get_data().view().into_scalar());
}

fn test_back_prop() {
    let x = Variable::new(arr0(0.5));
    let a = square(&x);
    let b = exp(&a);
    let y = square(&b);
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap().view().into_scalar());
}

fn test_numerical_diff() {
    let f = square;
    let x = Variable::new(arr0(2.0));
    let dy = numerical_diff(f, x, None);
    println!("dy = {:?}", dy.view().into_scalar());
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
