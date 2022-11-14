use crate::layer::{Layer, self};



#[derive(Debug, Clone)]
pub struct NN {
    n_inputs: usize,
    n_outputs: usize,
    layers: Vec<Layer>
}
impl NN {
    pub fn new(layer_sizes: Vec<usize>) -> NN {
        let mut layers: Vec<Layer> = vec![];

        let n_inputs = layer_sizes[0];
        let n_outputs = *layer_sizes.last().unwrap();

        for i in 1..layer_sizes.len() {
            layers.push(
                Layer::new(
                    layer_sizes[i-1],
                    layer_sizes[i]
                )
            );
        }

        NN { n_inputs, n_outputs, layers }
    }

    pub fn calc(&self, input: Vec<f32>) -> Vec<f32> {
        let mut a = input;
        for mut layer in &self.layers {
            a = layer.a(a);
        }
        a
    }

    pub fn train(&self, input: Vec<f32>, correct: Vec<f32>) {
        assert_eq!(self.n_outputs, correct.len(), "Invalid correct vec len");

        // calculate output of nn
        let output = self.calc(input);

        // calculate cost vector c of this run:
        let mut c: f32 = 0.0;
        for i in 0..self.n_outputs {
            c += ( output[i] - correct[i] ).powf(2.0);
        }

        //update all the layers
        for í in (1..10).rev() {
            //Do stuff
        }
        

    }
}