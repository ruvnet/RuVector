use ruvector_e_means::{fit, Config, IvfPartition, ProductQuantizer};

fn main() {
    let mut checks = 0;
    let mut check = |ok: bool| { assert!(ok); checks += 1; };
    let data = [0.0f32, 2.0, 4.0, 6.0, 8.0, 10.0];
    let config = Config { clusters: 1, epochs: 2, batch_size: Some(2), ..Config::default() };
    let model = fit(&data, 2, &config).unwrap();
    check(model.centroids == [4.0, 6.0]);
    check(model.assignments == [0, 0, 0]);
    check(model.predict(&data).unwrap() == [0, 0, 0]);
    check(model.mse.is_finite() && model.mse >= 0.0);
    check(model.updates == 4);
    check(fit(&data, 2, &config).unwrap().centroids == model.centroids);
    check(fit(&[f32::NAN], 1, &config).is_err());
    let pq = ProductQuantizer::fit(&data, 2, 2, &config).unwrap();
    check(pq.encode(&data).unwrap() == [0, 0, 0, 0, 0, 0]);
    check(pq.decode(&[0, 0]).unwrap() == [4.0, 6.0]);
    check(pq.decode(&[255, 0]).is_err());
    let ivf = IvfPartition::fit(&data, 2, &config).unwrap();
    check(ivf.posting(0).unwrap() == [0, 1, 2]);
    check(ivf.posting(1).is_err());
    println!("E_MEANS_PUBLIC_CHECKS={checks}");
}
