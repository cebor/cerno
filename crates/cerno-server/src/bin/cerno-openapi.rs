//! Prints the OpenAPI document.
//!
//! `spec/openapi.json` is generated from this, and `spec_is_current` in the server's test suite
//! fails when the checked-in copy drifts from the code. The SDKs are written against the spec,
//! so a silent drift would be a silent API break in three languages at once.

use cerno_server::ApiDoc;
use utoipa::OpenApi;

fn main() {
    println!(
        "{}",
        ApiDoc::openapi()
            .to_pretty_json()
            .expect("the derived document always serialises")
    );
}
