// Struct を callable にするには nightly の機能が必要 (Fn trait)
// https://stackoverflow.com/questions/42859330/how-do-i-make-a-struct-callable
#![feature(unboxed_closures)]
#![feature(fn_traits)]

// 構造体に関数を入れる
// https://stackoverflow.com/questions/27831944/how-do-i-store-a-closure-in-a-struct-in-rust

// 構造は本質的に doubly linked list なので、ノードの格納は Rc<RefCell<...>> に
// https://blog.ymgyt.io/entry/2019/08/17/013313
// https://gist.github.com/matey-jack/3e19b6370c6f7036a9119b79a82098ca
use std::cell::RefCell;
use std::rc::Rc;
// use std::fmt;


fn main() {
    test_numerical_diff();
    test_forward_prop();
    test_back_prop();
    test_back_prop_auto();
}

fn test_forward_prop() {
    let mut f = Square::new();
    let mut g = Exp::new();
    let mut h = Square::new();
    let x = Variable::new(0.5);
    let a = f(&x);
    let b = g(&a);
    let y = h(&b);
    println!("y = {:?}", y.get_data());
}

fn test_back_prop() {
    let mut f = Square::new();
    let mut g = Exp::new();
    let mut h = Square::new();
    let x = Variable::new(0.5);
    let a = f(&x);
    let b = g(&a);
    let y = h(&b);

    y.set_grad(1.0);
    b.set_grad(h.backward(y.get_grad().unwrap()));
    a.set_grad(g.backward(b.get_grad().unwrap()));
    x.set_grad(f.backward(a.get_grad().unwrap()));
    println!("x.grad = {:?}", x.get_grad().unwrap());
}

fn test_back_prop_auto() {
    let mut f = Square::new();
    let mut g = Exp::new();
    let mut h = Square::new();
    let x = Variable::new(0.5);
    let a = f(&x);
    let b = g(&a);
    let y = h(&b);

    y.set_grad(1.0);
    y.backward();
    println!("x.grad = {:?}", x.get_grad().unwrap());
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
    fn backward(&self) {
        if let Some(creator) = &self.creator {
            let mut funcs = vec![Rc::clone(creator)];
            while let Some(f) = funcs.pop() {
                if let (Some(x), Some(y)) = (&f.borrow().input, &f.borrow().output) {
                    x.borrow_mut().grad = y.borrow().grad.map(|x_| f.borrow().backward(x_));
                    x.borrow().creator.as_ref().map(|c| funcs.push(Rc::clone(c)));
                } else {
                    panic!("backward: input/output of creator not found");
                }
            }
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
    fn clone_cell(&self) -> Rc<RefCell<VariableCell>> {
        Rc::clone(&self.inner)
    }
    fn get_data(&self) -> Data {
        self.inner.borrow().data
    }
    fn get_grad(&self) -> Option<Data> {
        self.inner.borrow().grad
    }
    fn set_grad(&self, grad: Data) {
        self.inner.borrow_mut().grad = Some(grad);
    }
    fn set_creator(&self, func: Rc<RefCell<FunctionCell>>) {
        self.inner.borrow_mut().creator = Some(func);
    }
    fn backward(&self) {
        self.inner.borrow().backward()
    }
}

struct FunctionCell {
    input: Option<Rc<RefCell<VariableCell>>>,
    output: Option<Rc<RefCell<VariableCell>>>,
    backward: fn(Data, Data) -> Data,
}

impl FunctionCell {
    fn new(backward: fn(Data, Data) -> Data) -> Self {
        Self {
            input: None,
            output: None,
            backward: backward,
        }
    }
    fn cons(&mut self, input: &Variable, func: Rc<RefCell<FunctionCell>>, forward: fn(Data) -> Data) -> Variable {
        let x = input.get_data();
        let y = forward(x);
        let output = Variable::new(y);
        output.set_creator(func);
        self.input = Some(input.clone_cell());
        self.output = Some(output.clone_cell());
        output
    }
    fn backward(&self, gy: Data) -> Data {
        let x = self.input.as_ref().unwrap().borrow().data;
        (self.backward)(x, gy)
    }
}

struct Square {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Square {
    fn new() -> Self {
        Square {
            inner: Rc::new(RefCell::new(FunctionCell::new(Self::backward_body))),
        }
    }
    fn call(&self, input: &Variable) -> Variable {
        self.inner.borrow_mut().cons(input, Rc::clone(&self.inner), Self::forward_body)
    }
    fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(gy)
    }
    fn forward_body(x: Data) -> Data {
        x * x
    }
    fn backward_body(x: Data, gy: Data) -> Data {
        2.0 * x * gy
    }
}
impl FnOnce<(&Variable,)> for Square {
    type Output = Variable;
    extern "rust-call" fn call_once(self, _args: (&Variable,)) -> Variable {
        panic!("Square cannot be called as FnOnce")
    }
}
impl FnMut<(&Variable,)> for Square {
    // type Output = Variable
    extern "rust-call" fn call_mut(&mut self, args: (&Variable,)) -> Variable {
        self.call(&args.0)
    }
}

struct Exp {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Exp {
    fn new() -> Self {
        Exp {
            inner: Rc::new(RefCell::new(FunctionCell::new(Self::backward_body))),
        }
    }
    fn call(&self, input: &Variable) -> Variable {
        self.inner.borrow_mut().cons(input, Rc::clone(&self.inner), Self::forward_body)
    }
    fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(gy)
    }
    fn forward_body(x: Data) -> Data {
        x.exp()
    }
    fn backward_body(x: Data, gy: Data) -> Data {
        x.exp() * gy
    }
}
impl FnOnce<(&Variable,)> for Exp {
    type Output = Variable;
    extern "rust-call" fn call_once(self, _args: (&Variable,)) -> Variable {
        panic!("Exp cannot be called as FnOnce")
    }
}
impl FnMut<(&Variable,)> for Exp {
    // type Output = Variable
    extern "rust-call" fn call_mut(&mut self, args: (&Variable,)) -> Variable {
        self.call(&args.0)
    }
}

fn numerical_diff(f: &mut Square, x: Variable, eps: Option<Data>) -> Data {
    let e = eps.unwrap_or(1e-4);
    let x0 = Variable::new(x.get_data() - e);
    let x1 = Variable::new(x.get_data() + e);
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1.get_data() - y0.get_data()) / (2.0 * e)
}
