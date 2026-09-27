use agl_types::MutationScope;
fn main() {
    let permitted = MutationScope::ApplicationCode.auto_promotable();
    assert!(!permitted, "Application code must not bypass review");
    println!("{{\"scope\":\"ApplicationCode\",\"automatic_deployment_permitted\":{},\"delivery\":\"PR review only\",\"firmware_fitness_evaluated\":false}}", permitted);
}
