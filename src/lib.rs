use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

type Link<T> = Rc<RefCell<Node<T>>>;

#[derive(Debug)]
struct Node<T> {
    value: T,
    prev: Option<Link<T>>,
    next: Option<Link<T>>,
}

impl<T> Node<T> {
    fn new(value: T) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            value,
            prev: None,
            next: None,
        }))
    }
}

#[derive(Default)]
pub struct LinkedList<T> {
    head: Option<Link<T>>,
    tail: Option<Link<T>>,
    length: usize,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            length: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn append(&mut self, v: T) {
        let node = Node::new(v);
        match self.tail.take() {
            Some(old_tail) => {
                old_tail.borrow_mut().next = Some(Rc::clone(&node));
                node.borrow_mut().prev = Some(old_tail);
            }
            None => {
                // first element
                debug_assert_eq!(self.len(), 0);
                self.head = Some(Rc::clone(&node));
            }
        }

        self.tail = Some(node);
        self.length += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        match self.tail.take() {
            Some(tail) => {
                if let Some(prev) = tail.borrow_mut().prev.take() {
                    prev.borrow_mut().next = None;
                    self.tail = Some(prev);
                } else {
                    // we take last element
                    debug_assert_eq!(self.len(), 1);
                    self.head = None;
                }
                self.length -= 1;
                let v = Rc::try_unwrap(tail) // Rc<RefCell<Node<T>> -> RefCell<Node<T>>
                    .ok() // Result<RefCell<Node<T>>, Rc<RefCell<Node<T>>>> -> Option<RefCell<Node<T>>>
                    .expect("Failed to Rc::try_unwrap tail node") // RefCell<Node<T>>
                    .into_inner() // RefCell<Node<T>> -> Node<T>
                    .value;
                Some(v)
            }
            None => None,
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            current: if self.len() == 0 {
                None
            } else {
                Some(Rc::clone(&self.head.as_ref().unwrap()))
            },
        }
    }
}

impl<T: fmt::Display + Clone> fmt::Debug for LinkedList<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let iter = self.iter();
        write!(f, "{{ head")?;
        for v in iter {
            write!(f, " -> {}", v)?;
        }
        write!(f, " }}")
    }
}

impl<T: Clone> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Iter<T> {
    current: Option<Link<T>>,
}

impl<T: Clone> Iterator for Iter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.current.take() {
            None => None,
            Some(curr) => {
                let curr = curr.borrow();
                let v = curr.value.clone();
                match curr.next {
                    None => {
                        self.current = None;
                    }
                    Some(ref next) => {
                        self.current = Some(Rc::clone(next));
                    }
                }
                Some(v)
            }
        }
    }
}

impl<T: Clone> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<T> {
        match self.current.take() {
            None => None,
            Some(curr) => {
                let curr = curr.borrow();
                match curr.prev {
                    None => {
                        self.current = None;
                        None
                    }
                    Some(ref prev) => {
                        self.current = Some(Rc::clone(prev));
                        Some(prev.borrow().value.clone())
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug() {
        let mut list: LinkedList<i32> = LinkedList::new();
        list.append(1);
        list.append(2);
        assert_eq!(format!("{:?}", list), "{ head -> 1 -> 2 }");
    }

    #[test]
    fn empty_length() {
        let list: LinkedList<i32> = LinkedList::new();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn append() {
        let mut list: LinkedList<i32> = LinkedList::new();
        list.append(10);
        list.append(20);
        list.append(30);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn pop() {
        let mut list: LinkedList<i32> = LinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        assert_eq!(list.pop(), Some(3));
        assert_eq!(list.pop(), Some(2));
        assert_eq!(list.pop(), Some(1));
        assert_eq!(list.pop(), None);
    }

    #[test]
    fn iterator() {
        let mut list: LinkedList<i32> = LinkedList::new();
        (0..100).for_each(|n| list.append(n));

        let mut iter = list.iter();
        (0..100).for_each(|n| assert_eq!(iter.next(), Some(n)));
        assert_eq!(iter.next(), None);

        let mut iter = list.iter();
        (0..100).for_each(|n| assert_eq!(iter.next(), Some(n)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn reverse() {
        let mut list: LinkedList<i32> = LinkedList::new();
        (0..10).for_each(|n| list.append(n));

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next_back(), Some(2));
        assert_eq!(iter.next_back(), Some(1));
        assert_eq!(iter.next_back(), Some(0));
        assert_eq!(iter.next_back(), None);
    }
}