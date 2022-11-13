use std::vec;

use activation_functions::f32::sigmoid;
use rand::Rng;

fn main() {
    println!("Hello, world!");
    let nn = NN::new(vec![2, 2]);
    dbg!(&nn);

    let input = vec![0.2, 0.3];
    dbg!("{}", &nn.calc(input));
}

#[derive(Debug, Clone)]
struct NN {
    layers: Vec<Layer>
}
impl NN {
    fn new(layer_sizes: Vec<usize>) -> NN {
        let mut layers: Vec<Layer> = vec![];

        for i in 1..layer_sizes.len() {
            layers.push(
                Layer::new(
                    layer_sizes[i-1],
                    layer_sizes[i]
                )
            );
        }

        NN { layers: layers }
    }

    fn calc(&self, input: Vec<f32>) -> Vec<f32> {
        let mut a = input;
        for layer in &self.layers {
            a = layer.a(a);
        }
        a
    }
}


#[derive(Debug, Clone)]
struct Layer {
    size: usize,
    w: Vec<Vec<f32>>,
    b: Vec<f32>
}
impl Layer {
    fn new(nodes_in_prev_layer: usize, nodes_in_this_layer: usize) -> Layer {
        let mut rng = rand::thread_rng();

        Layer {
            size: nodes_in_this_layer,
            w: vec![
                vec![
                    rng.gen_range(0.0..10.0);
                    nodes_in_prev_layer
                ];
                nodes_in_this_layer
            ],
            b: vec![
                sigmoid(rand::thread_rng().gen_range(-1..1) as f32);
                nodes_in_this_layer
            ]
        }
    }

    fn a(&self, prev_layer_a: Vec<f32>) -> Vec<f32> {
        let mut a: Vec<f32> = vec![0.0; self.size];
        for i in 0..self.size {
            let cur_w = &self.w[i];
            let cur_b      = &self.b[i];

            let mut sum: f32 = 0.0;
            for j in 0..prev_layer_a.len() {
                sum += &cur_w[j] * &prev_layer_a[j];
            }

            a[i] = sigmoid(sum + cur_b);
        }

        a
    }
}