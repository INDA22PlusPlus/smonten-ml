// IMPORT MODULES
mod layer;
mod nn;
use nn::NN;
/////////////////


use mnist::*;
use ndarray::prelude::*;





fn main() {

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("/Users/seb/Documents/KTH/03/INDA/repos/smonten-ml/ml/src/data_sets/mnist")
        .finalize();
 
    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    // let s = format!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));
    // println!("s = {}", train_data.slice(s![image_num, .., ..]));

    for j in 0..10 {
        let mut i = 0;
        for x in train_data.slice(s![j, .., ..]) {
            let symbol = match (x * 10.0).round() / 10.0 {
                0.0 => " ",
                0.1 => " ",
                0.2 => " ",
                0.3 => ".",
                0.4 => "-",
                0.5 => "i",
                0.6 => "x",
                0.7 => "n",
                0.8 => "m",
                0.9 => "H",
                1.0 => "M",
                o => panic!("unexpected value: {}", o)
            };
            print!("{}", symbol);
            if i % 28 == 0 {
                println!()
            }
            i += 1;
        }
        println!();
    }
 
    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );
 
    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
 
    let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    
    let mut nn = NN::new(vec![784, 30, 10]);
    let step = 200;
    for k in 0..4 {
        let start = k*step;
        let stop = (k+1)*step;

        
        for j in start..stop {
            let mut input = vec![0.0; 784];
            let mut i = 0;
            for x in train_data.slice(s![j, .., ..]) {
                input[i] = *x;
                i += 1;
            }
            let correct_raw = format!("{:?}", train_labels.slice(s![image_num, ..]));
            let correct_idx = correct_raw[1..4].parse::<f32>().unwrap();
            let mut correct = vec![0.0; 10];
            correct[correct_idx as usize] = 1.0;
            // dbg!(correct);
            nn.train(input, correct);
        }
        nn.update();
    }

    println!("result!__________________");
    let j = 2000;
    let mut input = vec![0.0; 784];
    let mut i = 0;
    for x in train_data.slice(s![j, .., ..]) {
        input[i] = *x;
        i += 1;
    }
    let correct_raw = format!("{:?}", train_labels.slice(s![image_num, ..]));
    let correct_idx = correct_raw[1..4].parse::<f32>().unwrap();
    dbg!(correct_idx as usize);
    println!("{:?}", nn.calc(input));



}
