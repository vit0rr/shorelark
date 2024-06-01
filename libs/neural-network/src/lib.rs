use approx::assert_relative_eq;
use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub input_neurons: usize,
    pub output_neurons: usize,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].output_neurons, layers[1].input_neurons))
            .collect();

        Self { layers }
    }

    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

#[derive(Debug)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);

        Self { weights, bias }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_relative_eq;
//     use rand::SeedableRng;
//     use rand_chacha::ChaCha8Rng;

//     #[test]
//     fn random() {
//         // Because we always use the same seed, our `rng` in here will
//         // always return the same set of values
//         let mut rng = ChaCha8Rng::from_seed(Default::default());
//         let neuron = Neuron::random(&mut rng, 4);

//         assert_relative_eq!(neuron.bias, -0.6255188);

//         assert_relative_eq!(
//             neuron.weights.as_slice(),
//             [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
//         );
//     }
// }

#[test]
fn propagate() {
    let neuron = Neuron {
        bias: 0.5,
        weights: vec![-0.3, 0.8],
    };

    // Ensures `.max()` (our ReLU) works:
    assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

    // `0.5` and `1.0` chosen by a fair dice roll:
    assert_relative_eq!(
        neuron.propagate(&[0.5, 1.0]),
        (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
    );
}
