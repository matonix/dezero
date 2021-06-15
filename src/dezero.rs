// Struct を callable にするには nightly の機能が必要 (Fn trait)
// https://stackoverflow.com/questions/42859330/how-do-i-make-a-struct-callable
#![feature(unboxed_closures)]
#![feature(fn_traits)]

// 構造は本質的に doubly linked list なので、ノードの格納は Rc<RefCell<...>> に
// https://blog.ymgyt.io/entry/2019/08/17/013313
// https://gist.github.com/matey-jack/3e19b6370c6f7036a9119b79a82098ca
use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
// use std::fmt;

pub type Data = Array0<f64>;

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
                    x.borrow_mut().grad = y.borrow().grad.clone().map(|x_| f.borrow().backward(x_));
                    x.borrow()
                        .creator
                        .as_ref()
                        .map(|c| funcs.push(Rc::clone(c)));
                } else {
                    panic!("backward: input/output of creator not found");
                }
            }
        }
    }
}

pub struct Variable {
    inner: Rc<RefCell<VariableCell>>,
}
impl Variable {
    pub fn new(data: Data) -> Variable {
        Variable {
            inner: Rc::new(RefCell::new(VariableCell::new(data))),
        }
    }
    pub fn get_data(&self) -> Data {
        self.inner.borrow().data.clone()
    }
    pub fn get_grad(&self) -> Option<Data> {
        self.inner.borrow().grad.clone()
    }
    pub fn set_grad(&self, grad: Data) {
        self.inner.borrow_mut().grad = Some(grad);
    }
    pub fn backward(&self) {
        self.set_grad(Array::ones(self.get_data().raw_dim()));
        self.inner.borrow().backward()
    }
}

// 構造体に関数を入れる
// https://stackoverflow.com/questions/27831944/how-do-i-store-a-closure-in-a-struct-in-rust
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
    fn cons(
        &mut self,
        input: &Variable,
        func: Rc<RefCell<FunctionCell>>,
        forward: fn(Data) -> Data,
    ) -> Variable {
        let x = input.get_data();
        let y = forward(x);
        let output = Variable::new(y);
        output.inner.borrow_mut().creator = Some(func);
        self.input = Some(Rc::clone(&input.inner));
        self.output = Some(Rc::clone(&output.inner));
        output
    }
    fn backward(&self, gy: Data) -> Data {
        let x = self.input.as_ref().unwrap().borrow().data.clone();
        (self.backward)(x, gy)
    }
}
pub struct Square {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Square {
    pub fn new() -> Self {
        Square {
            inner: Rc::new(RefCell::new(FunctionCell::new(Self::backward_body))),
        }
    }
    pub fn call(&self, input: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(input, Rc::clone(&self.inner), Self::forward_body)
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(gy)
    }
    fn forward_body(x: Data) -> Data {
        x.map(|v| v * v)
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
pub fn square(x: &Variable) -> Variable {
    let f = Square::new();
    f.call(x)
}
pub struct Exp {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Exp {
    pub fn new() -> Self {
        Exp {
            inner: Rc::new(RefCell::new(FunctionCell::new(Self::backward_body))),
        }
    }
    pub fn call(&self, input: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(input, Rc::clone(&self.inner), Self::forward_body)
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(gy)
    }
    fn forward_body(x: Data) -> Data {
        x.mapv(f64::exp)
    }
    fn backward_body(x: Data, gy: Data) -> Data {
        x.mapv(f64::exp) * gy
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
pub fn exp(x: &Variable) -> Variable {
    let f = Exp::new();
    f.call(x)
}

pub fn numerical_diff(f: &mut Square, x: Variable, eps: Option<Data>) -> Data {
    let e = eps.unwrap_or(arr0(1e-4));
    let x0 = Variable::new(x.get_data() - e.clone());
    let x1 = Variable::new(x.get_data() + e.clone());
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1.get_data() - y0.get_data()) / (2.0 * e)
}
