//! The cerno engine: three typed primitives on top of one token of model output.
//!
//! A Noul, a Choice and a Score are the same operation wearing different names. Each offers the
//! model a handful of lettered options and reads the probability distribution over the *first*
//! token it would generate. That is one forward pass, no matter how many options there are —
//! which is what makes the whole thing fast, and what separates it from asking a model to write
//! an answer and then parsing the prose back out.
//!
//! ```text
//!   question ──> options ──> labels A,B,C ──> prompt ──> host ──> distribution
//!                                                                       │
//!            typed answer <── calibrate <── fold variants, floor <──────┘
//! ```
//!
//! The pieces are split so each can be tested on its own: [`labels`] owns the alphabet,
//! [`prompt`] owns the wording, [`math`] owns the arithmetic, and [`engine`] joins them.

pub mod engine;
pub mod labels;
pub mod math;
pub mod prompt;

pub use engine::{Engine, EngineError};
