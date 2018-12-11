#include <memory>
#include <iostream>
#include <string>
#include "cuda_common.h"
#include "NvInfer.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

namespace nvi = nvinfer1;

namespace lczero {
namespace {

class Logger : public nvi::ILogger {
    void log(Severity severity, const char* msg) override {
        std::cout << msg << std::endl;
    }
} gLogger;


class TensorRTNetwork : public Network {
private:
    nvi::IBuilder* builder;
    nvi::ICudaEngine* engine;
    int max_batch_size;
public:
    TensorRTNetwork(const WeightsFile& weights, const OptionsDict options);
    virtual ~TensorRTNetwork();

    std::unique_ptr<NetworkComputation> NewComputation() override;
};

nvi::Weights vector_to_weights(FloatVector const& vector) {
    return {nvi::DataType::kFLOAT, vector.data(), static_cast<int64_t>(vector.size())};
}

nvi::ITensor* name_and_get_output(nvi::ILayer* layer, std::string name) {
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

nvi::ITensor* process_bn(FloatVector bn_means, FloatVector bn_stddivs,
                         FloatVector bn_gammas, FloatVector bn_betas,
                         nvi::ITensor* input, nvi::INetworkDefinition* network,
                         std::string name = "") {
    auto num_channels = input->getDimensions().d[0];
    nvi::Dims3 dims_channel{num_channels, 1, 1};
    nvi::DataType dt = nvi::DataType::kFLOAT;

    auto bn_means_c = name_and_get_output(network->addConstant(dims_channel, vector_to_weights(bn_means)), name + "/constant_mean");
    auto bn_stddivs_c = name_and_get_output(network->addConstant(dims_channel, vector_to_weights(bn_stddivs)), name + "/constant_stddiv");

    input = name_and_get_output(network->addElementWise(*input, *bn_means_c, nvi::ElementWiseOperation::kMIN), name + "/mean_sub");
    input = name_and_get_output(network->addElementWise(*input, *bn_stddivs_c, nvi::ElementWiseOperation::kDIV), name + "/std_div");

    if(bn_betas.size() > 0) {
        auto bn_betas_c = network->addConstant(dims_channel, vector_to_weights(bn_betas))->getOutput(0);
        input = network->addElementWise(*input, *bn_betas_c, nvi::ElementWiseOperation::kSUM)->getOutput(0);
    }

    if(bn_gammas.size() > 0) {
        auto bn_gammas_c = network->addConstant(dims_channel, vector_to_weights(bn_gammas))->getOutput(0);
        input = network->addElementWise(*input, *bn_gammas_c, nvi::ElementWiseOperation::kPROD)->getOutput(0);
    }

    return input;
}

nvi::ITensor* process_convblock(const lczero::LegacyWeights::ConvBlock& conv_desc,
                                int numOutputMaps, nvi::ITensor* input, nvi::INetworkDefinition* network,
                                int size = 3, std::string name = "") {
    if(numOutputMaps == -1) {
        numOutputMaps = conv_desc.bn_means.size();
    }

    auto conv = network->addConvolution(*input, numOutputMaps, {size, size},
        vector_to_weights(conv_desc.weights),
        vector_to_weights(conv_desc.biases));
    
    conv->setPadding({(size - 1) / 2, (size - 1) / 2});
    conv->setName(name.c_str());

    return process_bn(conv_desc.bn_means, conv_desc.bn_stddivs,
                      conv_desc.bn_gammas, conv_desc.bn_betas,
                      conv->getOutput(0), network, name + "/bn");
}

nvi::ITensor* process_residual(const lczero::LegacyWeights::Residual& desc, nvi::ITensor* input, nvi::INetworkDefinition* network, std::string name = "") {
    int num_filters = input->getDimensions().d[0];
    auto current = input;

    current = process_convblock(desc.conv1, num_filters, current, network, 3, name + "/conv1");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kRELU), name + "/activation1");
    current = process_convblock(desc.conv2, num_filters, current, network, 3, name + "/conv2");
    current = name_and_get_output(network->addElementWise(*input, *current, nvi::ElementWiseOperation::kSUM), name + "/residual");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kRELU), name + "/activation2");

    return current;
}

nvi::ITensor* process_policy_head(const LegacyWeights::ConvBlock& conv,
                                  FloatVector ip_pol_w, FloatVector ip_pol_b,
                                  nvi::ITensor* input, nvi::INetworkDefinition* network) {
    auto current = input;
    current = process_convblock(conv, 32, current, network, 1, "policy/conv");
    current = name_and_get_output(network->addFullyConnected(
        *current, ip_pol_b.size(), vector_to_weights(ip_pol_w), vector_to_weights(ip_pol_b)), "policy/fc");
    
    current = name_and_get_output(network->addSoftMax(*current), "policy/softmax");

    return current;
}

nvi::ITensor* process_value_head(
        const LegacyWeights::ConvBlock& conv,
        FloatVector ip1_val_w,
        FloatVector ip1_val_b,
        FloatVector ip2_val_w,
        FloatVector ip2_val_b,
        nvi::ITensor* input,
        nvi::INetworkDefinition* network) {

    auto current = input;
    current = process_convblock(conv, 32, current, network, 1, "value/conv");
    current = name_and_get_output(network->addFullyConnected(*current, 128, vector_to_weights(ip1_val_w), vector_to_weights(ip1_val_b)), "value/fc1");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kRELU), "value/activation1");
    current = name_and_get_output(network->addFullyConnected(*current, 1, vector_to_weights(ip2_val_w), vector_to_weights(ip2_val_b)), "value/fc2");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kTANH), "value/output");

    return current;
}

TensorRTNetwork::TensorRTNetwork(const WeightsFile& file, const OptionsDict options)
    : builder(nullptr), engine(nullptr), max_batch_size(0) {

    LegacyWeights weights(file.weights());
    max_batch_size = options.GetOrDefault<int>("max_batch", 256);

    builder = nvi::createInferBuilder(gLogger);
    nvi::INetworkDefinition* network = builder->createNetwork();

    nvi::ITensor* current = network->addInput("input", nvi::DataType::kFLOAT, nvi::Dims3(112, 8, 8));
    current = process_convblock(weights.input, 256, current, network, 3, "initial_conv");

    // compute residual tower
    int counter = 1;
    for(auto&& block : weights.residual) {
        current = process_residual(block, current, network, "residual_block/" + std::to_string(counter));
        counter += 1;
    }

    // policy head
    auto output_pol = process_policy_head(weights.policy, weights.ip_pol_w, weights.ip_pol_b, current, network);
    output_pol->setName("output_policy");
    
    // value head
    auto output_val = process_value_head(
        weights.value, weights.ip1_val_w, weights.ip1_val_b, weights.ip2_val_w, weights.ip2_val_b,
        current, network);
    output_val->setName("output_value");
    
    network->markOutput(*output_pol);
    network->markOutput(*output_val);

    builder->setMaxBatchSize(max_batch_size);
    builder->setMaxWorkspaceSize(1 << 20);

    engine = builder->buildCudaEngine(*network);
    network->destroy();
}

TensorRTNetwork::~TensorRTNetwork() {
    if(engine) engine->destroy();
    if(builder) builder->destroy();
}

class TensorRTNetworkComputation : public NetworkComputation {
private:
    nvi::IExecutionContext* context;
    cudaStream_t stream;

    void* buffers[3];
    std::unique_ptr<float[]> input_buffer;
    std::unique_ptr<float[]> output_policy_buffer;
    std::unique_ptr<float[]> output_value_buffer;

    int input_index;
    int output_policy_index;
    int output_value_index;

    int in_batch;
    int max_batch_size;

    const int single_input_size = 112 * 8 * 8;
    const int single_output_pol_size = 1858;
public:
    TensorRTNetworkComputation(nvi::ICudaEngine* engine, int max_batch_size);
    virtual ~TensorRTNetworkComputation();

    void AddInput(InputPlanes&& input) override;
    // Do the computation.
    void ComputeBlocking() override;
    // Returns how many times AddInput() was called.
    int GetBatchSize() const override { return in_batch; }
    // Returns Q value of @sample.
    float GetQVal(int sample) const override { return output_value_buffer[sample]; }
    // Returns P value @move_id of @sample.
    float GetPVal(int sample, int move_id) const override { return output_policy_buffer[sample * single_output_pol_size + move_id]; }
};

TensorRTNetworkComputation::TensorRTNetworkComputation(nvi::ICudaEngine* engine, int max_batch_size)
    : context(nullptr), stream(0), buffers{nullptr, nullptr, nullptr},
      input_buffer(nullptr), output_policy_buffer(nullptr), output_value_buffer(nullptr),
      input_index(-1), output_policy_index(-1), output_value_index(-1),
      in_batch(0), max_batch_size(max_batch_size) {

    input_index = engine->getBindingIndex("input");
    output_policy_index = engine->getBindingIndex("output_policy");
    output_value_index = engine->getBindingIndex("output_value");
    context = engine->createExecutionContext();

    size_t input_size = max_batch_size * single_input_size;
    size_t output_pol_size = max_batch_size * single_output_pol_size;
    size_t output_val_size = max_batch_size;

    ReportCUDAErrors(cudaMalloc(&buffers[input_index], input_size * sizeof(float)));
    ReportCUDAErrors(cudaMalloc(&buffers[output_policy_index], output_pol_size * sizeof(float)));
    ReportCUDAErrors(cudaMalloc(&buffers[output_value_index], output_val_size * sizeof(float)));

    ReportCUDAErrors(cudaStreamCreate(&stream));

    input_buffer.reset(new float[input_size]);
    output_policy_buffer.reset(new float[output_pol_size]);
    output_value_buffer.reset(new float[output_val_size]);
}

// CPU version of plane-encoding.
// Eventually write custom plug-in for tensorrt and CUDA kernels.
void ExpandPlane(const InputPlane& plane, float* buffer) {
    const float value = plane.value;
    for (auto i = 0; i < 16; i++) {
        *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
    }
}

void TensorRTNetworkComputation::AddInput(InputPlanes&& input) {
    for(auto&& plane : input) {
        assert(in_batch < max_batch_size);
        ExpandPlane(plane, input_buffer.get() + in_batch * single_input_size);
        in_batch += 1;
    }
}

void TensorRTNetworkComputation::ComputeBlocking() {
    ReportCUDAErrors(cudaMemcpyAsync(buffers[input_index], input_buffer.get(),
        in_batch * single_input_size * sizeof(float),
        cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
    
    context->enqueue(in_batch, buffers, stream, nullptr);

    ReportCUDAErrors(cudaMemcpyAsync(output_policy_buffer.get(), buffers[output_policy_index],
        in_batch * single_output_pol_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    
    ReportCUDAErrors(cudaMemcpyAsync(output_value_buffer.get(), buffers[output_value_index],
        in_batch * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    
    ReportCUDAErrors(cudaStreamSynchronize(stream));

    in_batch = 0;
}


TensorRTNetworkComputation::~TensorRTNetworkComputation() {
    for(auto ptr : buffers) {
        if(ptr) cudaFree(ptr);
    }

    if(stream) ReportCUDAErrors(cudaStreamDestroy(stream));
    if(context) context->destroy();
}


std::unique_ptr<NetworkComputation> TensorRTNetwork::NewComputation() {
    return std::make_unique<TensorRTNetworkComputation>(engine, max_batch_size);
}

std::unique_ptr<Network> MakeTensorRTNetwork(const WeightsFile &weights, const OptionsDict &options) {
    return std::make_unique<TensorRTNetwork>(weights, options);
}

REGISTER_NETWORK("tensorrt", MakeTensorRTNetwork, 100);

}}