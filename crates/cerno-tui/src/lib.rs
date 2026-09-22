//! Terminal front end for the cerno service.
//!
//! The parts are split so that everything deciding *what* happens can be tested without a
//! terminal: [`draft`] parses the fields, [`app`] holds the state and builds the request through
//! the SDK's own builder, [`session`] remembers the last one, and [`render`] does the drawing.

pub mod app;
pub mod draft;
pub mod editor;
pub mod keys;
pub mod render;
pub mod session;
