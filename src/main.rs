use std::cell::RefCell;
use std::rc::{Rc, Weak};
// use std::fmt;

// 構造は本質的に doubly linked list
// https://blog.ymgyt.io/entry/2019/08/17/013313
// https://gist.github.com/matey-jack/3e19b6370c6f7036a9119b79a82098ca

// Struct を callable にするには nightly の機能が必要 (Fn trait)
// https://stackoverflow.com/questions/42859330/how-do-i-make-a-struct-callable

fn main() {
    test_numerical_diff();
    test_forward_prop();
    test_back_prop();
}

fn test_forward_prop() {
    let mut f = Square::new();
    let mut g = Exp::new();
    let mut h = Square::new();
    let x = Variable::new(0.5);
    let a = f.call(&x);
    let b = g.call(&a);
    let y = h.call(&b);
    println!("y = {:?}", y.get_data());
}

fn test_back_prop() {
    let mut f = Square::new();
    let mut g = Exp::new();
    let mut h = Square::new();
    let x = Variable::new(0.5);
    let a = f.call(&x);
    let b = g.call(&a);
    let y = h.call(&b);

    y.set_grad(1.0);
    b.set_grad(h.backward(y.get_grad()));
    a.set_grad(g.backward(b.get_grad()));
    x.set_grad(f.backward(a.get_grad()));
    println!("x = {:?}", x.get_grad());
}

fn test_numerical_diff() {
    let mut f = Square::new();
    let x = Variable::new(2.0);
    let dy = numerical_diff(&mut f, x, None);
    println!("dy = {:?}", dy);
}

type Data = f64;

struct VariableCell {
    data: Data,
    grad: Option<Data>,
    creator: Option<Rc<RefCell<FunctionCell>>>,
}
impl VariableCell {
    fn new(data: Data) -> VariableCell {
        VariableCell {
            data: data,
            grad: None,
            creator: None,
        }
    }
}

struct Variable {
    inner: Rc<RefCell<VariableCell>>,
}

impl Variable {
    fn new(data: Data) -> Variable {
        Variable {
            inner: Rc::new(RefCell::new(VariableCell::new(data))),
        }
    }
    fn get_data(&self) -> Data {
        self.inner.borrow().data
    }
    fn get_grad(&self) -> Data {
        self.inner.borrow().grad.unwrap()
    }
    fn clone_cell(&self) -> Rc<RefCell<VariableCell>> {
        Rc::clone(&self.inner)
    }
    fn set_grad(&self, grad: Data) {
        self.inner.borrow_mut().grad = Some(grad);
    }
    // fn backward(&self) {
    //     self.inner.borrow_mut().backward();
    // }
}

struct FunctionCell {
    forward: Box<dyn Fn(Data) -> Data>,
    input: Option<Rc<RefCell<VariableCell>>>,
    output: Option<Weak<RefCell<VariableCell>>>,
}

impl FunctionCell {
    fn new(forward: impl Fn(Data) -> Data + 'static) -> Self {
        Self {
            forward: Box::new(forward),
            input: None,
            output: None,
        }
    }
    fn cons(&mut self, input: &Variable) -> Variable {
        let x = input.get_data();
        self.input = Some(input.clone_cell());
        let y = (self.forward)(x);
        Variable::new(y)
    }
}

struct Square {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Square {
    fn new() -> Self {
        Square {
            inner: Rc::new(RefCell::new(FunctionCell::new(Square::forward))),
        }
    }
    fn call(&mut self, input: &Variable) -> Variable {
        self.inner.borrow_mut().cons(input)
    }
    fn forward(x: Data) -> Data {
        x * x
    }
    fn backward(&self, gy: Data) -> Data {
        let x = self.inner.borrow().input.as_ref().unwrap().borrow().data;
        2.0 * x * gy
    }
}

struct Exp {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Exp {
    fn new() -> Self {
        Exp {
            inner: Rc::new(RefCell::new(FunctionCell::new(Exp::forward))),
        }
    }
    fn call(&mut self, input: &Variable) -> Variable {
        self.inner.borrow_mut().cons(input)
    }
    fn forward(x: Data) -> Data {
        x.exp()
    }
    fn backward(&self, gy: Data) -> Data {
        let x = self.inner.borrow().input.as_ref().unwrap().borrow().data;
        x.exp() * gy
    }
}

fn numerical_diff(f: &mut Square, x: Variable, eps: Option<Data>) -> Data {
    let e = eps.unwrap_or(1e-4);
    let x0 = Variable::new(x.get_data() - e);
    let x1 = Variable::new(x.get_data() + e);
    let y0 = f.call(&x0);
    let y1 = f.call(&x1);
    (y1.get_data() - y0.get_data()) / (2.0 * e)
}