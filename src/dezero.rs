use ndarray::prelude::*;
// 構造は本質的に doubly linked list なので、ノードの格納は Rc<RefCell<...>> に
// https://blog.ymgyt.io/entry/2019/08/17/013313
// https://gist.github.com/matey-jack/3e19b6370c6f7036a9119b79a82098ca
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub type Data = Array0<f64>;

enum ForwardFn {
    OneOne(fn(&Data) -> Data),
    TwoOne(fn(&Data, &Data) -> Data),
}

enum BackwardFn {
    OneOneOne(fn(&Data, &Data) -> Data),
    OneOneTwo(fn(&Data, &Data) -> [Data; 2]),
}

#[derive(Clone)]
struct Candidate {
    priority: Reverse<usize>,
    id: usize,
    func: Rc<RefCell<FunctionCell>>,
}

impl Candidate {
    fn new(func: &Rc<RefCell<FunctionCell>>) -> Candidate {
        Candidate {
            priority: Reverse(func.borrow().generation), // max-heap
            id: func.borrow().object_id,
            func: Rc::clone(func),
        }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.id == other.id
    }
}

impl Eq for Candidate {}

impl Hash for Candidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct VariableCell {
    data: Data,
    grad: Option<Data>,
    creator: Option<Rc<RefCell<FunctionCell>>>,
    generation: usize,
    object_id: usize,
}
impl VariableCell {
    fn new(data: Data) -> VariableCell {
        VariableCell {
            data: data,
            grad: None,
            creator: None,
            generation: 0,
            object_id: 0,
        }
    }
    fn backward(&self) {
        if let Some(creator) = &self.creator {
            let mut funcs = BinaryHeap::<Candidate>::new();
            let mut seen_set = HashSet::new();
            fn add_func(
                f: &Rc<RefCell<FunctionCell>>,
                funcs: &mut BinaryHeap<Candidate>,
                seen_set: &mut HashSet<Candidate>,
            ) {
                let candidate = Candidate::new(&f);
                if !seen_set.contains(&candidate) {
                    seen_set.insert(candidate.clone());
                    funcs.push(candidate);
                }
            }
            add_func(creator, &mut funcs, &mut seen_set);
            while let Some(candidate) = funcs.pop() {
                let func = candidate.func;
                let gys = func
                    .borrow()
                    .outputs
                    .iter()
                    .flat_map(|y| y.borrow().grad.clone())
                    .collect::<Vec<_>>();
                let gxs = func.borrow().backward(gys);
                for (x, gx) in func.borrow().inputs.iter().zip(gxs) {
                    let gx_new = match &x.borrow().grad {
                        Some(v) => Some(v + gx),
                        None => Some(gx),
                    };
                    x.borrow_mut().grad = gx_new;
                    if let Some(creator) = &x.borrow().creator {
                        add_func(creator, &mut funcs, &mut seen_set);
                    }
                }
            }
        }
    }
    fn set_creator(&mut self, func_ref: &Rc<RefCell<FunctionCell>>, generation: usize) {
        self.generation = generation + 1;
        self.creator = Some(Rc::clone(func_ref));
    }
}

pub struct Variable {
    inner: Rc<RefCell<VariableCell>>,
}
impl Variable {
    pub fn new(data: Data) -> Variable {
        let v = Variable {
            inner: Rc::new(RefCell::new(VariableCell::new(data))),
        };
        // ポインタの値を object_id として扱う
        // https://users.rust-lang.org/t/object-identity/18402/11
        let v_ptr = Rc::into_raw(Rc::clone(&v.inner)) as *const _ as usize;
        v.inner.borrow_mut().object_id = v_ptr;
        v
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
    backward: BackwardFn,
    generation: usize,
    object_id: usize,
}
impl FunctionCell {
    fn new(backward: BackwardFn) -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            backward: backward,
            generation: 0,
            object_id: 0,
        }
    }
    fn cons(
        &mut self,
        inputs: Vec<&Variable>,
        func_ref: &Rc<RefCell<FunctionCell>>,
        forward: ForwardFn,
    ) -> Vec<Variable> {
        let xs = inputs
            .iter()
            .map(|input| input.get_data())
            .collect::<Vec<_>>();
        let ys = match forward {
            ForwardFn::OneOne(f) => vec![f(&xs[0])],
            ForwardFn::TwoOne(f) => vec![f(&xs[0], &xs[1])],
        };
        let outputs = ys
            .iter()
            .map(|y| Variable::new(y.clone()))
            .collect::<Vec<_>>();
        self.generation = inputs
            .iter()
            .map(|input| input.inner.borrow().generation)
            .max()
            .expect("generation");
        outputs.iter().for_each(|output| {
            output
                .inner
                .borrow_mut()
                .set_creator(func_ref, self.generation)
        });
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
            .map(|input| input.borrow().data.clone())
            .collect::<Vec<_>>();
        match self.backward {
            BackwardFn::OneOneOne(f) => vec![f(&xs[0], &gys[0])],
            BackwardFn::OneOneTwo(f) => f(&xs[0], &gys[0])
                .iter()
                .map(|x| x.clone())
                .collect::<Vec<_>>(),
        }
    }
}

pub struct Square {
    inner: Rc<RefCell<FunctionCell>>,
}
impl Square {
    pub fn new() -> Self {
        let f = Square {
            inner: Rc::new(RefCell::new(FunctionCell::new(BackwardFn::OneOneOne(
                Self::backward_body,
            )))),
        };
        // ポインタの値を object_id として扱う
        // https://users.rust-lang.org/t/object-identity/18402/11
        let f_ptr = Rc::into_raw(Rc::clone(&f.inner)) as *const _ as usize;
        f.inner.borrow_mut().object_id = f_ptr;
        f
    }
    pub fn call(&self, input: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(
                vec![input],
                &self.inner,
                ForwardFn::OneOne(Self::forward_body),
            )
            .pop()
            .unwrap()
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow().backward(vec![gy]).pop().unwrap()
    }
    fn forward_body(x: &Data) -> Data {
        x * x
    }
    fn backward_body(x: &Data, gy: &Data) -> Data {
        2.0 * x * gy
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
        let f = Exp {
            inner: Rc::new(RefCell::new(FunctionCell::new(BackwardFn::OneOneOne(
                Self::backward_body,
            )))),
        };
        // ポインタの値を object_id として扱う
        // https://users.rust-lang.org/t/object-identity/18402/11
        let f_ptr = Rc::into_raw(Rc::clone(&f.inner)) as *const _ as usize;
        f.inner.borrow_mut().object_id = f_ptr;
        f
    }
    pub fn call(&self, input: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(
                vec![input],
                &self.inner,
                ForwardFn::OneOne(Self::forward_body),
            )
            .pop()
            .unwrap()
    }
    pub fn backward(&self, gy: Data) -> Data {
        self.inner.borrow().backward(vec![gy]).pop().unwrap()
    }
    fn forward_body(x: &Data) -> Data {
        x.mapv(f64::exp)
    }
    fn backward_body(x: &Data, gy: &Data) -> Data {
        x.mapv(f64::exp) * gy
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
        let f = Add {
            inner: Rc::new(RefCell::new(FunctionCell::new(BackwardFn::OneOneTwo(
                Self::backward_body,
            )))),
        };
        // ポインタの値を object_id として扱う
        // https://users.rust-lang.org/t/object-identity/18402/11
        let f_ptr = Rc::into_raw(Rc::clone(&f.inner)) as *const _ as usize;
        f.inner.borrow_mut().object_id = f_ptr;
        f
    }
    pub fn call(&self, x: &Variable, y: &Variable) -> Variable {
        self.inner
            .borrow_mut()
            .cons(
                vec![x, y],
                &self.inner,
                ForwardFn::TwoOne(Self::forward_body),
            )
            .pop()
            .unwrap()
    }
    pub fn backward(&self, gy: Data) -> (Data, Data) {
        let mut gys = self.inner.borrow().backward(vec![gy]);
        (gys.pop().unwrap(), gys.pop().unwrap())
    }
    fn forward_body(x: &Data, y: &Data) -> Data {
        x + y
    }
    fn backward_body(_x: &Data, gy: &Data) -> [Data; 2] {
        [gy.clone(), gy.clone()]
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

impl fmt::Display for VariableCell {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        match &self.creator {
            Some(creator) => write!(w, "{}", creator.borrow())?,
            None => write!(w, "")?,
        }
        write!(w, "\nVar [ data = {}, grad = ", self.data)?;
        match &self.grad {
            Some(grad) => write!(w, "{}", grad)?,
            None => write!(w, "None")?,
        }
        write!(
            w,
            ", generation = {}, object_id = {} ]",
            self.generation, self.object_id
        )
    }
}

impl fmt::Display for FunctionCell {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        for input in &self.inputs {
            write!(w, "{}", input.borrow())?;
        }
        write!(
            w,
            "\nFun [ generation = {}, object_id = {} ]",
            self.generation, self.object_id
        )
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        write!(w, "{}", &self.inner.borrow())
    }
}
