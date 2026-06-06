#![allow(dead_code)]
#![allow(clippy::wildcard_dependencies, reason = "")]
#![allow(clippy::multiple_crate_versions, reason = "")]
#![allow(clippy::too_many_arguments, reason = "")]
#![allow(clippy::if_same_then_else, reason = "")]
#![allow(clippy::type_complexity, reason = "")]
#![allow(clippy::useless_format, reason = "")]
#![allow(clippy::absolute_paths, reason = "")]
#![allow(clippy::clone_on_ref_ptr, reason = "")]

mod multimodal_demo;
mod trainer_fixtures;
mod trainer_presentation;
mod trainer_walkthrough;

fn main() {
    println!("Multimodal brain demo");
    multimodal_demo::run_multimodal_demo();
    trainer_walkthrough::run_trainer_walkthrough();
}
