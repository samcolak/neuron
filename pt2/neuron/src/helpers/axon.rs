

use serde::{Serialize, Deserialize};


// Axon represents the sending end of a connection in a neural network. It connects one Dendrite to another and has a weight associated with it.

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axon {
    pub from: String,
    pub to: String,
    pub weight: i64,
}

impl Axon {
    
    pub fn new(from: String, to: String, weight: i64) -> Self {
        Self { from, to, weight }
    }

}