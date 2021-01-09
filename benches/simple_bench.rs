use criterion::{black_box, criterion_group, criterion_main, Criterion};
use treerite::{DMatrix, Predictor};

fn simple_bench(c: &mut Criterion) {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    let feat: Vec<f64> = vec![7.7, 2.8, 6.7, 2.];
    c.bench_function("simple", |b| {
        b.iter(|| model.predict_batch::<_, f64>(&DMatrix::from_slice(black_box(&feat)).unwrap(), false, false).unwrap())
    });
}

criterion_group!(benches, simple_bench);
criterion_main!(benches);
