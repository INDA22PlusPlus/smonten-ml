use activation_functions::f32::sigmoid;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Layer {
    size: usize,
    a: Vec<f32>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}
impl Layer {
    pub fn new(nodes_in_prev_layer: usize, nodes_in_this_layer: usize) -> Layer {
        // let rng = rand::thread_rng();

        let a = vec![0.0; nodes_in_this_layer];

        let w: Vec<Vec<f32>> = (0..nodes_in_this_layer).map(|_| 
            (0..nodes_in_prev_layer).map(|_| 
                rand::thread_rng().gen_range(-1.0..1.0)
            ).collect()
        ).collect();

        let b: Vec<f32> = (0..nodes_in_this_layer).map(|_|
            rand::thread_rng().gen_range(-1.0..1.0)
        ).collect();

        Layer {
            size: nodes_in_this_layer,
            a,
            w,
            b
        }


    }


    pub fn a(&mut self, prev_layer_a: Vec<f32>) -> Vec<f32> {
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
        self.a = a.clone();
        a
    }

    pub fn delC_dela(&self, delC_dela_next: Vec<f32>) -> Vec<f32> {
        let mut D_w: Vec<Vec<f32>> = vec![];
        let mut D_b: Vec<f32> = vec![];
        let mut delC_dela: Vec<f32> = vec![];

        for i in 0..self.size {

        }

        vec![0.0; self.size]
    }

}