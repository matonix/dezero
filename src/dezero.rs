use ndarray::prelude::*;
// 構造は本質的に doubly linked list なので、ノードの格納は Rc<RefCell<...>> に
// https://blog.ymgyt.io/entry/2019/08/17/013313
// https://gist.github.com/matey-jack/3e19b6370c6f7036a9119b79a82098ca
use std::cell::RefCell;
use std::rc::Rc;

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
                let gys = f
                    .borrow()
                    .outputs
                    .iter()
                    .flat_map(|y| y.borrow().grad.clone())
                    .collect::<Vec<_>>();
                let gxs = f.borrow().backward(gys);
                for (x, gx) in f.borrow().inputs.iter().zip(gxs) {
                    let gx_ = match &x.borrow().grad {
                        Some(v) => Some(v + gx),
                        None => Some(gx)
                    };
                    x.borrow_mut().grad = gx_;
                    x.borrow()
                        .creator
                        .as_ref()
                        .map(|c| funcs.push(Rc::clone(c)));
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
    pub fn clear_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }
}

// 構造体に関数を入れる
// https://stackoverflow.com/questions/27831944/how-do-i-store-a-closure-in-a-struct-in-rust
struct FunctionCell {
    inputs: Vec<Rc<RefCell<VariableCell>>>,
    outputs: Vec<Rc<RefCell<VariableCell>>>,
    backward: fn(Vec<Data>, Vec<Data>) -> Vec<Data>,
}
impl FunctionCell {
    fn new(backward: fn(Vec<Data>, Vec<Data>) -> Vec<Data>) -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            backward: backward,
        }
    }
    fn cons(
        &mut self,
        inputs: Vec<&Variable>,
        func: Rc<RefCell<FunctionCell>>,
        forward: fn(Vec<Data>) -> Vec<Data>,
    ) -> Vec<Variable> {
        let xs = inputs
            .iter()
            .map(|input| input.get_data())
            .collect::<Vec<_>>();
        let ys = forward(xs);
        let outputs = ys
            .iter()
            .map(|y| Variable::new(y.clone()))
            .collect::<Vec<_>>();
        outputs
            .iter()
            .for_each(|output| output.inner.borrow_mut().creator = Some(Rc::clone(&func)));
        self.inputs = inputs
            .iter()
            .map(|input| Rc::clone(&input.inner))
            .collect::<Vec<_>>();
        self.outputs = outputs
            .iter()
            .map(|output| Rc::clone(&output.inner))
            .collect::<Vec<_>>();
        outputs
    }
    fn backward(&self, gys: Vec<Data>) -> Vec<Data> {
        let xs = self
            .inputs
            .iter()
            .map(|input| input.as_ref().borrow().data.clone())
            .collect::<Vec<_>>();
        (self.backward)(xs, gys)
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
            .cons(vec![input], Rc::clone(&self.inner), Self::forward_body).pop().unwrap()
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(vec![gy]).pop().unwrap()
    }
    fn forward_body(x: Vec<Data>) -> Vec<Data> {
        vec![x[0].map(|v| v * v)]
    }
    fn backward_body(x: Vec<Data>, gy: Vec<Data>) -> Vec<Data> {
        vec![2.0 * &x[0] * &gy[0]]
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
            .cons(vec![input], Rc::clone(&self.inner), Self::forward_body).pop().unwrap()
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow_mut().backward(vec![gy]).pop().unwrap()
    }
    fn forward_body(x: Vec<Data>) -> Vec<Data> {
        vec![x[0].mapv(f64::exp)]
    }
    fn backward_body(x: Vec<Data>, gy: Vec<Data>) -> Vec<Data> {
        vec![x[0].mapv(f64::exp) * &gy[0]]
    }
}
pub fn exp(x: &Variable) -> Variable {
    let f = Exp::new();
    f.call(x)
}

pub struct Add {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Add {
    pub fn new() -> Self {
        Add {
            inner: Rc::new(RefCell::new(FunctionCell::new(Self::backward_body))),
        }
    }
    pub fn call(&self, x: &Variable, y: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(vec![x, y], Rc::clone(&self.inner), Self::forward_body).pop().unwrap()
    }
    pub fn backward(&self, gy: Data) -> (Data, Data) {
        let mut gys = self.inner.borrow_mut().backward(vec![gy]);
        (gys.pop().unwrap(), gys.pop().unwrap())
    }
    fn forward_body(xs: Vec<Data>) -> Vec<Data> {
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y = x0 + x1;
        vec!(y)
    }
    fn backward_body(_x: Vec<Data>, gy: Vec<Data>) -> Vec<Data> {
        let gy = &gy[0];
        vec![gy.clone() , gy.clone()]
    }
}
pub fn add(x: &Variable, y: &Variable) -> Variable {
    let f = Add::new();
    f.call(x, y)
}

pub fn numerical_diff(f: fn(&Variable) -> Variable, x: Variable, eps: Option<Data>) -> Data {
    let e = eps.unwrap_or(arr0(1e-4));
    let x0 = Variable::new(x.get_data() - e.clone());
    let x1 = Variable::new(x.get_data() + e.clone());
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1.get_data() - y0.get_data()) / (2.0 * e)
}
