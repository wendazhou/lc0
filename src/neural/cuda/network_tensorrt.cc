#include <memory>
#include <iostream>
#include <string>
#include <cstdio>
#include <mutex>
#include <queue>
#include "cuda_common.h"
#include "NvInfer.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

namespace nvi = nvinfer1;

namespace lczero {
namespace {

class Logger : public nvi::ILogger {
    void log(Severity severity, const char* msg) override {
        std::cout << msg << std::endl;
    }
} gLogger;


class TensorRTComputationResource {
private:
    void* buffers[3];
    float* input_buffer_;
    float* output_policy_buffer_;
    float* output_value_buffer_;

    cudaStream_t stream_;

    int max_batch_size_;
    int input_index;
    int output_policy_index;
    int output_value_index;
    int resource_id_;

    const int single_input_size = 112 * 8 * 8;
    const int single_output_pol_size = 1858;
public:
    TensorRTComputationResource(int max_batch_size, int input_index, int output_policy_index, int output_value_index, int resource_id = -1)
        : buffers{nullptr, nullptr, nullptr}, input_buffer_(nullptr), output_policy_buffer_(nullptr), output_value_buffer_(nullptr),
            stream_(0), max_batch_size_(max_batch_size), input_index(input_index),
            output_policy_index(output_policy_index), output_value_index(output_value_index), resource_id_(resource_id) {
        
        size_t input_size = max_batch_size * single_input_size;
        size_t output_pol_size = max_batch_size * single_output_pol_size;
        size_t output_val_size = max_batch_size;
        
        ReportCUDAErrors(cudaMalloc(&buffers[input_index], input_size * sizeof(float)));
        ReportCUDAErrors(cudaMalloc(&buffers[output_policy_index], output_pol_size * sizeof(float)));
        ReportCUDAErrors(cudaMalloc(&buffers[output_value_index], output_val_size * sizeof(float)));

        ReportCUDAErrors(cudaMallocHost(&input_buffer_, input_size * sizeof(float)));
        ReportCUDAErrors(cudaMallocHost(&output_policy_buffer_, output_pol_size * sizeof(float)));
        ReportCUDAErrors(cudaMallocHost(&output_value_buffer_, output_val_size * sizeof(float)));

        ReportCUDAErrors(cudaStreamCreate(&stream_));
        std::string name = "Network Compute Stream " + std::to_string(resource_id_);
        nvtxNameCudaStreamA(stream_, name.c_str());
    }

    ~TensorRTComputationResource() {
        for(auto ptr : buffers) {
            if(ptr) cudaFree(ptr);
        }

        if (input_buffer_) ReportCUDAErrors(cudaFreeHost(input_buffer_));
        if (output_policy_buffer_) ReportCUDAErrors(cudaFreeHost(output_policy_buffer_));
        if (output_value_buffer_) ReportCUDAErrors(cudaFreeHost(output_value_buffer_));

        if(stream_) ReportCUDAErrors(cudaStreamDestroy(stream_));
    }

    cudaStream_t& stream() { return stream_; }
    float* input_buffer() { return input_buffer_; }
    float* output_policy_buffer() { return output_policy_buffer_; }
    float* output_value_buffer() { return output_value_buffer_; }

    int max_batch_size() const { return max_batch_size_; }
    int resource_id() const { return resource_id_; }

    void EnqueueLoadData(int batch_size) {
        ReportCUDAErrors(cudaMemcpyAsync(buffers[input_index], input_buffer_,
            batch_size * single_input_size * sizeof(float),
            cudaMemcpyKind::cudaMemcpyHostToDevice, stream_));
    }

    void EnqueueUnloadResults(int batch_size) {
        ReportCUDAErrors(cudaMemcpyAsync(output_policy_buffer_, buffers[output_policy_index],
            batch_size * single_output_pol_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_));
        
        ReportCUDAErrors(cudaMemcpyAsync(output_value_buffer_, buffers[output_value_index],
            batch_size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_));
    }

    void EnqueueContextExecute(int batch_size, nvi::IExecutionContext* context) {
        context->enqueue(batch_size, buffers, stream_, nullptr);
    }

    void SynchronizeStream() {
        ReportCUDAErrors(cudaStreamSynchronize(stream_));
    }
};


class ComputationResourceCache {
    int max_batch_size_;
    int input_index_;
    int output_policy_index_;
    int output_value_index_;

    int total_resources;

    std::mutex mutex;
    std::queue<std::unique_ptr<TensorRTComputationResource>> resources;
public:
    ComputationResourceCache(int max_batch_size, int input_index, int output_policy_index, int output_value_index)
        : max_batch_size_(max_batch_size), input_index_(input_index),
          output_policy_index_(output_policy_index), output_value_index_(output_value_index),
          total_resources(0), mutex(), resources()
    {
    }

    std::unique_ptr<TensorRTComputationResource> GetNewResource() {
        std::lock_guard<std::mutex> lock(mutex);

        if(resources.empty()) {
            total_resources += 1;
            return std::make_unique<TensorRTComputationResource>(max_batch_size_, input_index_,
                output_policy_index_, output_value_index_, total_resources);
        }

        std::unique_ptr<TensorRTComputationResource> result = std::move(resources.front());
        resources.pop();

        return result;
    }

    void ReleaseResource(std::unique_ptr<TensorRTComputationResource> resource) {
        std::lock_guard<std::mutex> lock(mutex);
        resources.emplace(std::move(resource));
    }
};


class TensorRTNetwork : public Network {
private:
    nvi::IRuntime* runtime;
    nvi::IBuilder* builder;
    nvi::ICudaEngine* engine;
    nvi::IExecutionContext* context;
    int max_batch_size;
    std::unique_ptr<ComputationResourceCache> resource_cache;
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

bool is_identity_elementwise(const FloatVector& values, float identity = 0) {
    return std::all_of(values.begin(), values.end(), [=](float x) { return x == identity; });
}

nvi::ITensor* process_bn(const FloatVector& bn_means, const FloatVector& bn_stddivs,
                         const FloatVector& bn_gammas,const FloatVector& bn_betas,
                         nvi::ITensor* input, nvi::INetworkDefinition* network,
                         std::string name = "") {
    auto num_channels = input->getDimensions().d[0];
    nvi::Dims3 dims_channel{num_channels, 1, 1};
    nvi::DataType dt = nvi::DataType::kFLOAT;

    if(!is_identity_elementwise(bn_means, 0.0)) {
        auto bn_means_c = name_and_get_output(network->addConstant(dims_channel, vector_to_weights(bn_means)), name + "/constant_mean");
        input = name_and_get_output(network->addElementWise(*input, *bn_means_c, nvi::ElementWiseOperation::kMIN), name + "/mean_sub");
    }

    if(!is_identity_elementwise(bn_stddivs, 1.0)) {
        auto bn_stddivs_c = name_and_get_output(network->addConstant(dims_channel, vector_to_weights(bn_stddivs)), name + "/constant_stddiv");
        input = name_and_get_output(network->addElementWise(*input, *bn_stddivs_c, nvi::ElementWiseOperation::kDIV), name + "/std_div");
    }

    if(!is_identity_elementwise(bn_betas, 0.0)) {
        auto bn_betas_c = network->addConstant(dims_channel, vector_to_weights(bn_betas))->getOutput(0);
        input = network->addElementWise(*input, *bn_betas_c, nvi::ElementWiseOperation::kSUM)->getOutput(0);
    }

    if(!is_identity_elementwise(bn_gammas, 1.0)) {
        auto bn_gammas_c = network->addConstant(dims_channel, vector_to_weights(bn_gammas))->getOutput(0);
        input = network->addElementWise(*input, *bn_gammas_c, nvi::ElementWiseOperation::kPROD)->getOutput(0);
    }

    return input;
}

void simplify_conv_block(LegacyWeights::ConvBlock& block, bool foldBNLayer = false) {
    const float epsilon = 1e-5f;

    // Compute reciprocal of std-dev from the variances (so that it can be
    // just multiplied).
    std::vector<float>& stddev = block.bn_stddivs;
    for (auto&& w : stddev) {
        w = 1.0f / std::sqrt(w + epsilon);
    }

    // Biases are not calculated and are typically zero but some networks
    // might still have non-zero biases. Move biases to batchnorm means to
    // make the output match without having to separately add the biases.
    for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
        block.bn_means[j] -= block.biases[j];
        block.biases[j] = 0.0f;
    }

    // Get rid of the BN layer by adjusting weights and biases of the
    // convolution idea proposed by Henrik Forstn and first implemented in
    // leela go zero.
    if (foldBNLayer) {
        const int outputs = block.biases.size();
        const int channels = block.weights.size() / (outputs * 3 * 3);

        for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto i = 0; i < 9; i++) {
            block.weights[o * channels * 9 + c * 9 + i] *= block.bn_stddivs[o];
            }
        }

        block.bn_means[o] *= block.bn_stddivs[o];
        block.bn_stddivs[o] = 1.0f;

        // Move means to convolution biases.
        block.biases[o] = -block.bn_means[o];
        block.bn_means[o] = 0.0f;
        }
    }
}

nvi::ITensor* process_convblock(lczero::LegacyWeights::ConvBlock& conv_desc,
                                int numOutputMaps, nvi::ITensor* input, nvi::INetworkDefinition* network,
                                int size = 3, std::string name = "") {
    if(numOutputMaps == -1) {
        numOutputMaps = conv_desc.bn_means.size();
    }

    simplify_conv_block(conv_desc, true);

    auto conv = network->addConvolution(*input, numOutputMaps, {size, size},
        vector_to_weights(conv_desc.weights),
        vector_to_weights(conv_desc.biases));
    
    conv->setPadding({(size - 1) / 2, (size - 1) / 2});
    conv->setName(name.c_str());

    return process_bn(conv_desc.bn_means, conv_desc.bn_stddivs,
                      conv_desc.bn_gammas, conv_desc.bn_betas,
                      conv->getOutput(0), network, name + "/bn");
}

nvi::ITensor* process_residual(lczero::LegacyWeights::Residual& desc, nvi::ITensor* input, nvi::INetworkDefinition* network, std::string name = "") {
    int num_filters = input->getDimensions().d[0];
    auto current = input;

    current = process_convblock(desc.conv1, num_filters, current, network, 3, name + "/conv1");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kRELU), name + "/activation1");
    current = process_convblock(desc.conv2, num_filters, current, network, 3, name + "/conv2");
    current = name_and_get_output(network->addElementWise(*input, *current, nvi::ElementWiseOperation::kSUM), name + "/residual");
    current = name_and_get_output(network->addActivation(*current, nvi::ActivationType::kRELU), name + "/activation2");

    return current;
}

nvi::ITensor* process_policy_head(LegacyWeights::ConvBlock& conv,
                                  const FloatVector& ip_pol_w, const FloatVector& ip_pol_b,
                                  nvi::ITensor* input, nvi::INetworkDefinition* network) {
    auto current = input;
    current = process_convblock(conv, 32, current, network, 1, "policy/conv");
    current = name_and_get_output(network->addFullyConnected(
        *current, ip_pol_b.size(), vector_to_weights(ip_pol_w), vector_to_weights(ip_pol_b)), "policy/fc");
    
    current = name_and_get_output(network->addSoftMax(*current), "policy/softmax");

    return current;
}


nvi::ITensor* process_value_head(
        LegacyWeights::ConvBlock& conv,
        const FloatVector& ip1_val_w,
        const FloatVector& ip1_val_b,
        const FloatVector& ip2_val_w,
        const FloatVector& ip2_val_b,
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

void save_serialized_model(nvi::IHostMemory* model) {
    auto file = fopen("serialized_model.dat", "wb");
    size_t size = model->size();

    fwrite(&size, sizeof(size), 1, file);
    fwrite(model->data(), 1, size, file);

    fclose(file);
}

std::tuple<nvi::ICudaEngine*, nvi::IRuntime*> load_serialized_model(nvi::ILogger& logger) {
    auto file = fopen("serialized_model.dat", "rb");

    if(!file) {
        return {nullptr, nullptr};
    }

    size_t size;
    fread(&size, sizeof(size), 1, file);

    char* buffer = new char[size];

    fread(buffer, 1, size, file);
    fclose(file);

    nvi::IRuntime* runtime = nvi::createInferRuntime(logger);
    nvi::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;

    return std::make_tuple(engine, runtime);
}

std::tuple<nvi::ICudaEngine*, nvi::IBuilder*> create_model(nvi::ILogger& logger, LegacyWeights& weights, int max_batch_size) {
    nvi::IBuilder* builder = nvi::createInferBuilder(logger);
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
    builder->setMaxWorkspaceSize(1 << 25);

    nvi::ICudaEngine* engine = builder->buildCudaEngine(*network);
    network->destroy();

    return std::make_tuple(engine, builder);
}

TensorRTNetwork::TensorRTNetwork(const WeightsFile& file, const OptionsDict options)
    : runtime(nullptr), builder(nullptr), engine(nullptr), max_batch_size(0) {

    max_batch_size = options.GetOrDefault<int>("max_batch", 256);

    std::tie(engine, runtime) = load_serialized_model(gLogger);

    if(!engine) {
        LegacyWeights weights(file.weights());
        std::tie(engine, builder) = create_model(gLogger, weights, max_batch_size);

        // serialize created model
        nvi::IHostMemory* serializedModel = engine->serialize();
        save_serialized_model(serializedModel);
        serializedModel->destroy();
    }

    context = engine->createExecutionContext();

    int input_index = engine->getBindingIndex("input");
    int output_policy_index = engine->getBindingIndex("output_policy");
    int output_value_index = engine->getBindingIndex("output_value");

    resource_cache = std::make_unique<ComputationResourceCache>(max_batch_size, input_index, output_policy_index, output_value_index);
}

TensorRTNetwork::~TensorRTNetwork() {
    if(context) context->destroy();
    if(engine) engine->destroy();
    if(builder) builder->destroy();
    if(runtime) runtime->destroy();
}

class TensorRTNetworkComputation : public NetworkComputation {
private:
    nvi::IExecutionContext* context;
    ComputationResourceCache* cache;
    std::unique_ptr<TensorRTComputationResource> resource;

    int in_batch;
    bool computation_done;

    const int single_input_size = 112 * 8 * 8;
    const int single_output_pol_size = 1858;
public:
    TensorRTNetworkComputation(nvi::IExecutionContext* context, ComputationResourceCache* resource_cache);
    virtual ~TensorRTNetworkComputation();

    void AddInput(InputPlanes&& input) override;
    // Do the computation.
    void ComputeBlocking() override;
    // Returns how many times AddInput() was called.
    int GetBatchSize() const override { return in_batch; }
    // Returns Q value of @sample.
    float GetQVal(int sample) const override { assert(computation_done); return resource->output_value_buffer()[sample]; }
    // Returns P value @move_id of @sample.
    float GetPVal(int sample, int move_id) const override { assert(computation_done); return resource->output_policy_buffer()[sample * single_output_pol_size + move_id]; }
};

TensorRTNetworkComputation::TensorRTNetworkComputation(nvi::IExecutionContext* context, ComputationResourceCache* resource_cache)
    : context(context), cache(resource_cache), in_batch(0), computation_done(false), resource(resource_cache->GetNewResource()) { }

// CPU version of plane-encoding.
// Eventually write custom plug-in for tensorrt and CUDA kernels.
void EncodePlanes(const InputPlanes& sample, float* buffer) {
  for (const InputPlane& plane : sample) {
    const float value = plane.value;
    for (auto i = 0; i < 64; i++)
      *(buffer++) = (plane.mask & (((uint64_t)1) << i)) != 0 ? value : 0;
  }
}

void TensorRTNetworkComputation::AddInput(InputPlanes&& input) {
    assert(in_batch < resource->max_batch_size());
    assert(input.size() == 112);

    EncodePlanes(input, resource->input_buffer() + in_batch * single_input_size);
    in_batch += 1;
}

void TensorRTNetworkComputation::ComputeBlocking() {
    std::string range_name = "NetworkEvaluation" + std::to_string(resource->resource_id());
    auto range_id = nvtxRangeStartA(range_name.c_str());

    resource->EnqueueLoadData(in_batch);
    resource->EnqueueContextExecute(in_batch, context);
    resource->EnqueueUnloadResults(in_batch);
    resource->SynchronizeStream();

    nvtxRangeEnd(range_id);

    in_batch = 0;
    computation_done = true;
}


TensorRTNetworkComputation::~TensorRTNetworkComputation() {
    cache->ReleaseResource(std::move(resource));
}


std::unique_ptr<NetworkComputation> TensorRTNetwork::NewComputation() {
    return std::make_unique<TensorRTNetworkComputation>(context, resource_cache.get());
}

std::unique_ptr<Network> MakeTensorRTNetwork(const WeightsFile &weights, const OptionsDict &options) {
    return std::make_unique<TensorRTNetwork>(weights, options);
}

REGISTER_NETWORK("tensorrt", MakeTensorRTNetwork, 100);

}}