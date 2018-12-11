#include <memory>
#include <iostream>
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

class ExpandPlanesPlugin : public nvi::IPluginV2 {
public:
    virtual ~ExpandPlanesPlugin() {}

    int getNbOutputs() const override { return 1; }

    nvi::Dims getOutputDimensions(int index, const nvi::Dims* inputs, int nbInputDims) override;
    bool supportsFormat(nvi::DataType datatype, nvi::PluginFormat format) override {
        return (datatype == nvi::DataType::kFLOAT) && (format == nvi::PluginFormat::kNCHW);
    }
};

class TensorRTNetwork : public Network {
private:
    nvi::IBuilder* builder;
    nvi::ICudaEngine* engine;
public:
    TensorRTNetwork(const WeightsFile& weights, const OptionsDict options);
    virtual ~TensorRTNetwork();

    std::unique_ptr<NetworkComputation> NewComputation() override;
};

nvi::Weights vector_to_weights(FloatVector const& vector) {
    return {nvi::DataType::kFLOAT, vector.data(), vector.size()};
}

nvi::ITensor* process_bn(FloatVector bn_means, FloatVector bn_stddivs,
                         FloatVector bn_gammas, FloatVector bn_betas,
                         nvi::ITensor* input, nvi::INetworkDefinition* network) {
    auto num_channels = input->getDimensions().d[0];
    nvi::Dims3 dims_channel{num_channels, 1, 1};
    nvi::DataType dt = nvi::DataType::kFLOAT;

    auto bn_means_c = network->addConstant(dims_channel, vector_to_weights(bn_means))->getOutput(0);
    auto bn_stddivs_c = network->addConstant(dims_channel, vector_to_weights(bn_stddivs))->getOutput(0);
    auto bn_gammas_c = network->addConstant(dims_channel, vector_to_weights(bn_gammas))->getOutput(0);
    auto bn_betas_c = network->addConstant(dims_channel, vector_to_weights(bn_betas))->getOutput(0);

    input = network->addElementWise(*input, *bn_means_c, nvi::ElementWiseOperation::kMIN)->getOutput(0);
    input = network->addElementWise(*input, *bn_stddivs_c, nvi::ElementWiseOperation::kDIV)->getOutput(0);
    input = network->addElementWise(*input, *bn_betas_c, nvi::ElementWiseOperation::kSUM)->getOutput(0);
    input = network->addElementWise(*input, *bn_gammas_c, nvi::ElementWiseOperation::kPROD)->getOutput(0);

    return input;
}

nvi::ITensor* process_convblock(const lczero::LegacyWeights::ConvBlock& conv_desc,
                       int numOutputMaps, nvi::ITensor* input, nvi::INetworkDefinition* network) {
    
    auto conv = network->addConvolution(*input, numOutputMaps, {3, 3},
        vector_to_weights(conv_desc.weights),
        vector_to_weights(conv_desc.biases));

    return process_bn(conv_desc.bn_means, conv_desc.bn_stddivs,
                      conv_desc.bn_gammas, conv_desc.bn_betas,
                      conv->getOutput(0), network);
}

nvi::ITensor* process_residual(const lczero::LegacyWeights::Residual& desc, nvi::ITensor* input, nvi::INetworkDefinition* network) {
    int num_filters = input->getDimensions().d[0];
    auto current = input;

    current = process_convblock(desc.conv1, num_filters, current, network);
    current = network->addActivation(*current, nvi::ActivationType::kRELU)->getOutput(0);
    current = process_convblock(desc.conv2, num_filters, current, network);
    current = network->addElementWise(*input, *current, nvi::ElementWiseOperation::kSUM)->getOutput(0);
    current = network->addActivation(*current, nvi::ActivationType::kRELU)->getOutput(0);

    return current;
}

nvi::ITensor* process_policy_head(const LegacyWeights::ConvBlock& conv,
                                  FloatVector ip_pol_w, FloatVector ip_pol_b,
                                  nvi::ITensor* input, nvi::INetworkDefinition* network) {
    auto current = input;
    current = process_convblock(conv, 32, current, network);
    current = network->addFullyConnected(
        *current, 1858, vector_to_weights(ip_pol_w), vector_to_weights(ip_pol_b))->getOutput(0);
    
    current = network->addSoftMax(*current)->getOutput(0);

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
    current = process_convblock(conv, 32, current, network);
    current = network->addFullyConnected(*current, 128, vector_to_weights(ip1_val_w), vector_to_weights(ip1_val_b))->getOutput(0);
    current = network->addActivation(*current, nvi::ActivationType::kRELU)->getOutput(0);
    current = network->addFullyConnected(*current, 1, vector_to_weights(ip2_val_w), vector_to_weights(ip2_val_b))->getOutput(0);
    current = network->addActivation(*current, nvi::ActivationType::kTANH)->getOutput(0);

    return current;
}

TensorRTNetwork::TensorRTNetwork(const WeightsFile& file, const OptionsDict options)
    : builder(nullptr), engine(nullptr) {
    LegacyWeights weights(file.weights());

    builder = nvi::createInferBuilder(gLogger);
    nvi::INetworkDefinition* network = builder->createNetwork();

    nvi::ITensor* current = network->addInput("input", nvi::DataType::kFLOAT, nvi::Dims3(112, 8, 8));
    current = process_convblock(weights.input, 256, current, network);

    // compute residual tower
    for(auto&& block : weights.residual) {
        current = process_residual(block, current, network);
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

    auto max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
    builder->setMaxBatchSize(max_batch_size_);
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
    int GetBatchSize() const override;
    // Returns Q value of @sample.
    float GetQVal(int sample) const override;
    // Returns P value @move_id of @sample.
    float GetPVal(int sample, int move_id) const override;
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

}}