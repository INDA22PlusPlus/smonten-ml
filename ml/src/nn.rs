use crate::layer::{Layer, self};



#[derive(Debug, Clone)]
pub struct NN {
    n_inputs: usize,
    n_outputs: usize,
    n_layers: usize,
    layers: Vec<Layer>
}
impl NN {
    pub fn new(layer_sizes: Vec<usize>) -> NN {
        let mut layers: Vec<Layer> = vec![];

        let n_inputs = layer_sizes[0];
        let n_outputs = *layer_sizes.last().unwrap();
        let n_layers = layer_sizes.len()-1; // we are not counting input layer here

        for i in 1..layer_sizes.len() {
            layers.push(
                Layer::new(
                    layer_sizes[i-1],
                    layer_sizes[i]
                )
            );
        }

        NN { n_inputs, n_outputs, n_layers, layers }
    }

    pub fn calc(& mut self, input: Vec<f32>) -> Vec<f32> {
        let mut a = input;
        for i in 0..self.n_layers {
            a = self.layers[i].a(a);
        }
        a
    }

    pub fn train(&mut self, input: Vec<f32>, correct: Vec<f32>) -> f32 {
        assert_eq!(self.n_outputs, correct.len(), "Invalid correct vec len");

        // calculate output of nn
        let output = self.calc(input.clone());

        // calculate cost vector c of this run:
        let mut c: f32 = 0.0;
        for i in 0..self.n_outputs {
            c += ( output[i] - correct[i] ).powf(2.0);
        }

        let a_last_layer = self.layers.last().unwrap().get_cur_activation();
        let mut delC_dela_last_layer: Vec<Vec<f32>> = vec![vec![0.0]; self.n_outputs];
        for i in 0..self.n_outputs {
            let delC_delai_last_layer = 2.0 * (a_last_layer[i] - correct[i]) * 1.0;
            delC_dela_last_layer[i] = vec![delC_delai_last_layer];
        }


        //update all the layers
        let mut delC_dela_cur_layer = delC_dela_last_layer;
        for i in (0..self.n_layers).rev() {
            let prev_layer_a = match i {
                0 => input.clone(),
                _ => self.layers[i-1].get_cur_activation()
            };
            delC_dela_cur_layer = self.layers[i].delC_dela_prev_layer(
                delC_dela_cur_layer,
                prev_layer_a,
            );
        }

        return c; // the error

    }

    pub fn update(&mut self) {
        // calculate average partial derivative and update w and b
        for i in (0..self.n_layers).rev() {
            self.layers[i].update();
        }
    }
}