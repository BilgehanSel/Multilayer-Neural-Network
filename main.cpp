// Bilgehan Sel - 11/10/2017
// Neural Network class that has adjustable input, output and hidden neuron size;
// New NeuralNetwork variable is gotten by
// nn(input_count, hidden_neuron_count, output_count)
// to train the network,
// inside a for loop (specify how many times you want it to continue)
// simply write nn.Train(train_data_3d_vector)
// and to test it
// write nn.Test(test_data_3d_vector)

// inspired by
// http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/


#include <iostream>
#include <time.h>
#include <math.h>
#include <vector>

class NeuralNetwork{
public:
	NeuralNetwork(unsigned const& input_size, unsigned const& hidden_size, unsigned const& output_size) {
		// Initializing weights vector
		std::vector<std::vector<double>> temp_hidden_weights(hidden_size, std::vector<double>(input_size));
		std::vector<std::vector<double>> temp_output_weights(output_size, std::vector<double>(hidden_size));
		// Initializing Random weights for hidden weights
		for (unsigned i = 0; i != temp_hidden_weights.size(); i++) {
			for (unsigned j = 0; j != temp_hidden_weights[0].size(); j++) {
				temp_hidden_weights[i][j] = double(rand() % 100 + 1) / 1000;
			}
		}
		// Initializing Random weights for output weights
		for (unsigned i = 0; i != temp_output_weights.size(); i++) {
			for (unsigned j = 0; j != temp_output_weights[0].size(); j++) {
				temp_output_weights[i][j] = double(rand() % 100 + 1) / 1000;
			}
		}
		// push back temp weights to hidden_weights;
		weights.push_back(temp_hidden_weights);
		weights.push_back(temp_output_weights);
		// ------------------------------------------------------------------------------------------------------ \\

		// Initializing a vector
		a.push_back(std::vector<double>(input_size));
		a.push_back(std::vector<double>(hidden_size));
		a.push_back(std::vector<double>(output_size));

		// Initializing z vector
		z.push_back(std::vector<double>(1)); // the first layer we don't need z
		z.push_back(std::vector<double>(hidden_size));
		z.push_back(std::vector<double>(output_size));

		// Initializing bias_weights
		bias_weights.push_back(std::vector<double>(hidden_size));
		bias_weights.push_back(std::vector<double>(output_size));

		// Initializing error_weights
		error.push_back(std::vector<double>(hidden_size));
		error.push_back(std::vector<double>(output_size));

		// Initializing y
		y.resize(output_size);

		// Initializng learning_rate
		learning_rate = 0.001;

		// Setting bias to 1
		bias = 1;
	}

	void Train(std::vector<std::vector<std::vector<double>>>& train_data) {
		// Loop Through train_data examples
		for (unsigned i = 0; i != train_data.size(); i++) {
			// Setting xi to ai for the input layer
			for (unsigned j = 0; j != a[0].size(); j++) {
				a[0][j] = train_data[i][0][j];
			}
			// Setting y
			for (unsigned j = 0; j != a[2].size(); j++) {
				y[j] = train_data[i][1][j];
			}
			// FeedForward first layer
			FeedForward(0);
			// Activate first layer
			Activation(1);

			// FeedForward second layer
			FeedForward(1);
			// Activate second layer
			Activation(2);
			J();
			BackPropagation();
		}
	}

	void Test(std::vector<std::vector<std::vector<double>>>& test_data) {
		// Loop Through test_data examples
		for (unsigned i = 0; i != test_data.size(); i++) {
			// Setting xi to ai for the input layer
			for (unsigned j = 0; j != a[0].size(); j++) {
				a[0][j] = test_data[i][0][j];
			}
			// Setting y
			for (unsigned j = 0; j != a[2].size(); j++) {
				y[j] = test_data[i][1][j];
			}
			// FeedForward first layer
			FeedForward(0);
			// Activate first layer
			Activation(1);

			// FeedForward second layer
			FeedForward(1);
			// Activate second layer
			Activation(2);

			std::cout << "Network's answer: " << a[2][0] << " || " << test_data[i][1][0] << " Real Answer." << std::endl;
		}
	}

private:
	// Cost Function
	void J() {
		for (unsigned i = 0; i != error[1].size(); i++) {
			//error[1][i] = (y[i] - a[2][i]);
			error[1][i] = (y[i] - a[2][i]) * a[2][i] * (1 - a[2][i]);
		}
	}

	// BackPropagation Algorithm
	void BackPropagation() {
		// Compute error[0] vector
		for (unsigned i = 0; i != error[0].size(); i++) {
			double sum = 0;
			for (unsigned j = 0; j != error[1].size(); j++) {
				sum += error[1][j] * weights[1][j][i];
			}
			error[0][i] = sum;
		}

		// Update weights[1]
		for (unsigned i = 0; i != weights[1].size(); i++) {
			for (unsigned j = 0; j != weights[1][i].size(); j++) {
				weights[1][i][j] = weights[1][i][j] + learning_rate * a[1][j] * error[1][i];
			}
			bias_weights[1][i] = bias_weights[1][i] + learning_rate * error[1][i];
		}

		// Update weights[0]
		for (unsigned i = 0; i != weights[0].size(); i++) {
			for (unsigned j = 0; j != weights[0][i].size(); j++) {
				weights[0][i][j] = weights[0][i][j] + learning_rate * a[0][j] * error[0][i];
			}
			bias_weights[0][i] = bias_weights[0][i] + learning_rate * error[0][i];
		}
	}

	void FeedForward(unsigned const& l) {
		std::vector<double> temp_z;
		for (unsigned i = 0; i != weights[l].size(); i++) {
			double sum = 0;
			for (unsigned j = 0; j != weights[l][i].size(); j++) {
				sum += weights[l][i][j] * a[l][j];
			}
			sum += bias_weights[l][i];
			temp_z.push_back(sum);
		}
		z[l + 1] = temp_z;
		a[l + 1] = z[l + 1];
	}

	void Activation(unsigned const& l) {
		// Sigmoid Function
		for (unsigned i = 0; i != a[l].size(); i++) {
			a[l][i] = 1 / (1 + std::exp(-z[l][i]));
		}
	}

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> bias_weights;
	// Activated output of the neurons
	std::vector<std::vector<double>> a;
	// Unactivated output of the neurons
	std::vector<std::vector<double>> z;
	std::vector<std::vector<double>> error;
	std::vector<double> y;
	double learning_rate;
	double bias;
};

int main() {
	std::srand(time(0));
	std::vector<std::vector<std::vector<double>>> train_data = { { { 5, 3 }, { 0 } }, { { 12, 5 }, { 1 } }, { { 3, 1 }, { 1 } }, { { 0, 1 }, { 0 } }, { { -2, -3 }, { 1 } }, { { -10, -4 }, { 0 } }, { { 2, 0 }, { 1 } }, { { 2.2, 1.1 }, { 1 } }, { { 6, 2 }, { 1 } }, { { 1.2, 0.2 }, { 1 } }, { { 4.8, 3.5 }, { 0 } }, { { -2.3, -1.5 }, { 1 } }, { { -1, -0.5 }, { 1 } }, { { 1.2, 0.8 }, { 0 } }, { { -9, -4 }, { 0 } }, { { 15, 6 }, { 1 } }, { { 3, 6 }, { 0 } }, { { 8, 4.5 }, { 0 } }, { { 5.7, 3 }, { 0 } } };
	std::vector<std::vector<std::vector<double>>> test_data = { { { 4, 1 }, { 1 } }, { { 5.5, 3 }, { 1 } }, { { 8, 7 }, { 0 } }, { { -9, -12 }, { 1 } } };
	NeuralNetwork nn(2, 20, 1);

	for (unsigned i = 0; i != 10000; i++) {
		nn.Train(train_data);
	}

	nn.Test(test_data);
}
