use activation_functions::f32::sigmoid;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Layer {
    size: usize,
    cur_a: Vec<f32>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    Dw_tot: Vec<Vec<f32>>,
    Db_tot: Vec<f32>,
    train_count: usize,

}
impl Layer {
    pub fn new(nodes_in_prev_layer: usize, nodes_in_this_layer: usize) -> Layer {
        // let rng = rand::thread_rng();

        let cur_a = vec![0.0; nodes_in_this_layer];

        let w: Vec<Vec<f32>> = (0..nodes_in_this_layer).map(|_| 
            (0..nodes_in_prev_layer).map(|_| 
                rand::thread_rng().gen_range(-1.0..1.0)
            ).collect()
        ).collect();

        let Dw_tot = vec![vec![0.0; nodes_in_prev_layer]; nodes_in_this_layer];

        let b: Vec<f32> = (0..nodes_in_this_layer).map(|_|
            rand::thread_rng().gen_range(-1.0..1.0)
        ).collect();

        let Db_tot = vec![0.0; nodes_in_this_layer];

        let train_count: usize = 0;

        Layer {
            size: nodes_in_this_layer,
            cur_a,
            w,
            b,
            Dw_tot,
            Db_tot,
            train_count
        }


    }


    pub fn a(&mut self, prev_layer_a: Vec<f32>) -> Vec<f32> {
        let mut activation: Vec<f32> = vec![0.0; self.size];

        for i in 0..self.size {
            let this_w = &self.w[i];
            let this_b      = &self.b[i];

            let mut z: f32 = 0.0;
            for j in 0..prev_layer_a.len() {
                z += &this_w[j] * &prev_layer_a[j];
            }
            z += this_b;

            activation[i] = sigmoid(z);
        }
        self.cur_a = activation.clone();
        return activation;
    }

    pub fn get_cur_activation(&self) -> Vec<f32> {
        self.cur_a.clone()
    }

    pub fn delC_dela_prev_layer(&mut self, delC_dela_this_layer: Vec<Vec<f32>>, prev_layer_a: Vec<f32>) -> Vec<Vec<f32>> {

        let mut delC_dela_prev_layer: Vec<Vec<f32>> = vec![vec![0.0; self.size]; prev_layer_a.len()];

        for i in 0..self.size {
            // sum together all the elements in each row of delC_dela_this_layer
            let sum_delC_delai = {
                let mut sum = 0.0;
                for k in 0..delC_dela_this_layer[i].len() {
                    sum += delC_dela_this_layer[i][k];
                }
                sum
            };
            let delai_delzi = sum_delC_delai * (1.0 - sum_delC_delai);

            // b
            let delzi_delbi = 1.0;
            let delC_delbi = sum_delC_delai * delai_delzi * delzi_delbi;
            self.Db_tot[i] += delC_delbi;
            // w
            for j in 0..prev_layer_a.len() {

                //save delC_dela_prev_layerij
                delC_dela_prev_layer[j][i] = sum_delC_delai * delai_delzi * self.w[i][j]; // obs reverse orders of i and j here

                //update wij
                let delzi_delwij = prev_layer_a[j];
                let delC_delwij = sum_delC_delai * delai_delzi * delzi_delwij;
                self.Dw_tot[i][j] +=  delC_delwij;
            }

        }

        self.train_count += 1;

        return delC_dela_prev_layer;
    }

    pub fn update(&mut self) {
        let scale = 0.2;
        
        
        for i in 0..self.size {
            // update b
            self.b[i] -= scale * self.Db_tot[i] / (self.train_count as f32);
            // reset Db_toti
            self.Db_tot[i] = 0.0;

            // update w
            for j in 0..self.w[0].len() {
                self.w[i][j] -= scale * self.Dw_tot[i][j] / (self.train_count as f32);
                // reset Dw_totij
                self.Dw_tot[i][j] = 0.0;
            }
        }
        // reset train count
        self.train_count = 0;
    }

}