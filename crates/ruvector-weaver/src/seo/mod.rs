//! Portfolio SEO / distribution layer (ADR-356): canonical topic taxonomy,
//! metadata audit, exact-diff proposals and the repo/package apply path.
//! Visibility comes from metadata, README cross-links built from REAL
//! dependency edges, shared topics and the ruv.io homepage, never from
//! adding dependencies.

pub mod apply;
pub mod audit;
pub mod baseline;
pub mod diff;
pub mod facts;
pub mod flow;
pub mod jsontext;
pub mod propose;
pub mod taxonomy;

pub const LINKS_START: &str = "<!-- weaver:seo:links:start -->";
pub const LINKS_END: &str = "<!-- weaver:seo:links:end -->";
pub const FOOTER_START: &str = "<!-- weaver:seo:footer:start -->";
pub const FOOTER_END: &str = "<!-- weaver:seo:footer:end -->";
pub const FLOW_START: &str = "<!-- weaver:seo:flow:start -->";
pub const FLOW_END: &str = "<!-- weaver:seo:flow:end -->";
