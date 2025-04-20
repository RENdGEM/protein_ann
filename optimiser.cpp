#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <numeric>
#include <chrono>

float clamp(float value, float min_val, float max_val)
{
    return std::max(min_val, std::min(value, max_val));
}

using namespace std;

struct ActivationFunction
{
    virtual float activate(float x) const = 0;
    virtual float derivative(float x) const = 0;
    virtual ~ActivationFunction() = default;
};

struct ReLU : ActivationFunction
{
    float activate(float x) const override { return x > 0 ? x : 0.01f * x; }
    float derivative(float x) const override { return x > 0 ? 1.0f : 0.01f; }
};

struct Identity : ActivationFunction
{
    float activate(float x) const override { return x; }
    float derivative(float x) const override { return 1.0f; }
};

struct Normalizer
{
    vector<float> mins, maxs;
    explicit Normalizer(const vector<vector<float>> &data)
    {
        size_t D = data[0].size();
        mins.assign(D, 1e9f);
        maxs.assign(D, -1e9f);
        for (const auto &row : data)
        {
            for (size_t i = 0; i < D; ++i)
            {
                mins[i] = min(mins[i], row[i]);
                maxs[i] = max(maxs[i], row[i]);
            }
        }
    }

    vector<float> normalize(const vector<float> &v) const
    {
        vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            float range = maxs[i] - mins[i];
            out[i] = (range == 0) ? 0.0f : (v[i] - mins[i]) / range;
        }
        return out;
    }

    vector<float> denormalize(const vector<float> &v) const
    {
        vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            out[i] = v[i] * (maxs[i] - mins[i]) + mins[i];
        }
        return out;
    }
};

struct Neuron
{
    vector<float> weights;
    float bias;
    float pre_activation;
    shared_ptr<ActivationFunction> act;

    Neuron(size_t inputs, shared_ptr<ActivationFunction> a) : act(a)
    {
        random_device rd;
        mt19937 gen(rd());
        float fan_in = static_cast<float>(inputs);
        float stddev = sqrt(2.0f / fan_in); // He initialization
        normal_distribution<float> dist(0.0f, stddev);

        weights.resize(inputs);
        for (auto &w : weights)
            w = dist(gen);
        bias = dist(gen);
    }

    float forward(const vector<float> &in)
    {
        pre_activation = inner_product(in.begin(), in.end(), weights.begin(), bias);
        return act->activate(pre_activation);
    }

    void update(const vector<float> &inputs, float delta, float lr)
    {
        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] -= lr * delta * inputs[i];
        }
        bias -= lr * delta;
    }
};

struct Layer
{
    vector<Neuron> neurons;
    vector<float> last_input;

    Layer(size_t num_neurons, size_t inputs, shared_ptr<ActivationFunction> a)
    {
        for (size_t i = 0; i < num_neurons; ++i)
            neurons.emplace_back(inputs, a);
    }

    vector<float> forward(const vector<float> &in)
    {
        last_input = in;
        vector<float> out;
        for (auto &n : neurons)
            out.push_back(n.forward(in));
        return out;
    }
};

class Network
{
    vector<Layer> layers;
    Normalizer input_norm, output_norm;
    float learning_rate = 0.01f;
    float momentum = 0.9f;

public:
    Network(size_t in, const vector<size_t> &hidden, size_t out,
            const vector<vector<float>> &X,
            const vector<vector<float>> &Y)
        : input_norm(X), output_norm(Y)
    {
        size_t prev_size = in;
        for (auto h : hidden)
        {
            layers.emplace_back(h, prev_size, make_shared<ReLU>());
            prev_size = h;
        }
        layers.emplace_back(out, prev_size, make_shared<Identity>());
    }

    vector<float> predict(const vector<float> &in)
    {
        auto normalized = input_norm.normalize(in);
        for (auto &layer : layers)
        {
            normalized = layer.forward(normalized);
        }
        return output_norm.denormalize(normalized);
    }

    void train(const vector<vector<float>> &X, const vector<vector<float>> &Y,
               int epochs = 10000, float lr = 0.01f)
    {
        learning_rate = lr;
        vector<vector<vector<float>>> weight_velocities;
        vector<vector<float>> bias_velocities;

        // Initialize momentum buffers
        for (const auto &layer : layers)
        {
            vector<vector<float>> layer_vel;
            for (const auto &neuron : layer.neurons)
            {
                layer_vel.push_back(vector<float>(neuron.weights.size(), 0.0f));
            }
            weight_velocities.push_back(layer_vel);
            bias_velocities.push_back(vector<float>(layer.neurons.size(), 0.0f));
        }

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float total_loss = 0;
            for (size_t i = 0; i < X.size(); ++i)
            {
                auto x_norm = input_norm.normalize(X[i]);
                auto y_norm = output_norm.normalize(Y[i]);

                // Forward pass
                vector<vector<float>> layer_outputs;
                layer_outputs.push_back(x_norm);
                for (auto &layer : layers)
                {
                    layer_outputs.push_back(layer.forward(layer_outputs.back()));
                }

                // Calculate loss
                auto &output = layer_outputs.back();
                vector<float> errors(output.size());
                for (size_t j = 0; j < output.size(); ++j)
                {
                    errors[j] = output[j] - y_norm[j];
                    total_loss += errors[j] * errors[j];
                }

                // Backward pass
                for (int l = layers.size() - 1; l >= 0; --l)
                {
                    auto &layer = layers[l];
                    vector<float> grad_prev(layer.last_input.size(), 0.0f);

                    for (size_t n = 0; n < layer.neurons.size(); ++n)
                    {
                        auto &neuron = layer.neurons[n];
                        float delta = errors[n] * neuron.act->derivative(neuron.pre_activation);

                        // Update weights with momentum
                        for (size_t w = 0; w < neuron.weights.size(); ++w)
                        {
                            float &velocity = weight_velocities[l][n][w];
                            velocity = momentum * velocity + learning_rate * delta * layer.last_input[w];
                            neuron.weights[w] -= velocity;
                            grad_prev[w] += neuron.weights[w] * delta;
                        }

                        // Update bias with momentum
                        float &b_velocity = bias_velocities[l][n];
                        b_velocity = momentum * b_velocity + learning_rate * delta;
                        neuron.bias -= b_velocity;
                    }

                    errors = std::move(grad_prev);
                }
            }

            if (epoch % 1000 == 0)
            {
                cout << "Epoch " << epoch << " Loss: " << total_loss / X.size() << endl;
            }
        }
    }
};

vector<float> maximize_output(Network &net, int iterations = 5000, float lr = 0.1f)
{
    vector<float> current_input = {9.0f, 15.0f, 120.0f}; // Center of input space
    float best_score = -INFINITY;
    vector<float> best_input;

    for (int iter = 0; iter < iterations; ++iter)
    {
        vector<float> grad(3, 0.0f);
        float eps = 0.001f;
        float base_score = 0.0f;

        auto base_output = net.predict(current_input);
        base_score = accumulate(base_output.begin(), base_output.end(), 0.0f);

        for (int i = 0; i < 3; ++i)
        {
            auto perturbed = current_input;
            perturbed[i] += eps;

            perturbed[i] = clamp(perturbed[i],
                                 (i == 0) ? 8.0f : (i == 1) ? 10.0f
                                                            : 60.0f,
                                 (i == 0) ? 10.0f : (i == 1) ? 20.0f
                                                             : 180.0f);

            auto output = net.predict(perturbed);
            float perturbed_score = accumulate(output.begin(), output.end(), 0.0f);
            grad[i] = (perturbed_score - base_score) / eps;
        }

        for (int i = 0; i < 3; ++i)
        {
            current_input[i] += lr * grad[i];
            current_input[i] = clamp(current_input[i],
                                     (i == 0) ? 8.0f : (i == 1) ? 10.0f
                                                                : 60.0f,
                                     (i == 0) ? 10.0f : (i == 1) ? 20.0f
                                                                 : 180.0f);
        }
        auto current_output = net.predict(current_input);
        float current_score = accumulate(current_output.begin(), current_output.end(), 0.0f);
        if (current_score > best_score)
        {
            best_score = current_score;
            best_input = current_input;
        }

        lr *= 0.999f;
    }

    return best_input;
}

int main()
{
    vector<vector<float>> X = {
        {8, 10, 120}, {10, 10, 120}, {8, 20, 120}, {10, 20, 120}, {8, 15, 60}, {10, 15, 60}, {8, 15, 180}, {10, 15, 180}, {9, 10, 60}, {9, 20, 60}, {9, 10, 180}, {9, 20, 180}, {9, 15, 120}};

    vector<vector<float>> Y = {
        {81.4075, 6.88799, 95.0233}, {92.4086, 56.1088, 69.108}, {85.7495, 6.85181, 92.5148}, {91.4249, 69.5573, 72.9006}, {75.1371, 5.62957, 90.1651}, {89.6936, 50.9223, 70.3451}, {78.5223, 7.72359, 92.4692}, {90.5473, 53.0204, 67.2352}, {90.384, 23.2539, 81.8382}, {91.3365, 26.1623, 85.0955}, {90.8792, 17.1941, 81.5964}, {93.7101, 38.8466, 88.016}, {92.6409, 30.8956, 84.1428}};

    Network net(3, {9, 12, 9}, 3, X, Y);

    net.train(X, Y, 10000, 0.005f);

    cout << "\nPredictions on training data:\n";
    for (size_t i = 0; i < X.size(); ++i)
    {
        auto pred = net.predict(X[i]);
        cout << "Input: ";
        for (auto v : X[i])
            cout << v << " ";
        cout << "| Predicted: ";
        for (auto v : pred)
            cout << v << " ";
        cout << "| Target: ";
        for (auto v : Y[i])
            cout << v << " ";
        cout << endl;
    }
    cout<<endl<<endl;
    // net.train(X, Y, 10000, 0.005f);

    // Find optimal input
    auto start = chrono::high_resolution_clock::now();
    vector<float> optimal_input = maximize_output(net, 5000, 0.1f);
    auto end = chrono::high_resolution_clock::now();

    // Display results
    auto output = net.predict(optimal_input);
    cout << "\nOptimized Input: ";
    for (auto v : optimal_input) cout << v << " ";
    
    cout << "\nPredicted Outputs: ";
    for (auto v : output) cout << v << " ";
    
    cout << "\nTotal Score: " << output[0] + output[1] + output[2];
    cout << "\nOptimization Time: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << "ms\n";
    return 0;
}