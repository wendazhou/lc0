#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include "NvInfer.h"
#include "cuda_common.h"
#include "network_tensorrt_definition.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

namespace nvi = nvinfer1;

namespace lczero {
namespace tensorrt_backend {
namespace {

class Logger : public nvi::ILogger {
  void log(Severity severity, const char* msg) override {
    std::cout << msg << std::endl;
  }
} gLogger;

class TensorRTComputationResource {
 private:
  nvi::IExecutionContext* context_;
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
  TensorRTComputationResource(nvi::ICudaEngine* engine, int resource_id = -1)
      : context_(nullptr),
        buffers{nullptr, nullptr, nullptr},
        input_buffer_(nullptr),
        output_policy_buffer_(nullptr),
        output_value_buffer_(nullptr),
        stream_(0),
        max_batch_size_(-1),
        input_index(-1),
        output_policy_index(-1),
        output_value_index(-1),
        resource_id_(resource_id) {
    max_batch_size_ = engine->getMaxBatchSize();

    input_index = engine->getBindingIndex(INPUT_BLOB_NAME);
    output_policy_index = engine->getBindingIndex(OUTPUT_POL_BLOB_NAME);
    output_value_index = engine->getBindingIndex(OUTPUT_VAL_BLOB_NAME);

    size_t input_size = max_batch_size_ * single_input_size;
    size_t output_pol_size = max_batch_size_ * single_output_pol_size;
    size_t output_val_size = max_batch_size_;

    ReportCUDAErrors(
        cudaMalloc(&buffers[input_index], input_size * sizeof(float)));
    ReportCUDAErrors(cudaMalloc(&buffers[output_policy_index],
                                output_pol_size * sizeof(float)));
    ReportCUDAErrors(cudaMalloc(&buffers[output_value_index],
                                output_val_size * sizeof(float)));

    ReportCUDAErrors(
        cudaMallocHost(&input_buffer_, input_size * sizeof(float)));
    ReportCUDAErrors(cudaMallocHost(&output_policy_buffer_,
                                    output_pol_size * sizeof(float)));
    ReportCUDAErrors(
        cudaMallocHost(&output_value_buffer_, output_val_size * sizeof(float)));

    ReportCUDAErrors(cudaStreamCreate(&stream_));
    std::string name = "Network Compute Stream " + std::to_string(resource_id_);
    nvtxNameCudaStreamA(stream_, name.c_str());

    context_ = engine->createExecutionContext();
  }

  ~TensorRTComputationResource() {
    for (auto ptr : buffers) {
      if (ptr) cudaFree(ptr);
    }

    if (input_buffer_) ReportCUDAErrors(cudaFreeHost(input_buffer_));
    if (output_policy_buffer_)
      ReportCUDAErrors(cudaFreeHost(output_policy_buffer_));
    if (output_value_buffer_)
      ReportCUDAErrors(cudaFreeHost(output_value_buffer_));

    if (stream_) ReportCUDAErrors(cudaStreamDestroy(stream_));
    context_->destroy();
  }

  cudaStream_t& stream() { return stream_; }
  float* input_buffer() { return input_buffer_; }
  float* output_policy_buffer() { return output_policy_buffer_; }
  float* output_value_buffer() { return output_value_buffer_; }

  int max_batch_size() const { return max_batch_size_; }
  int resource_id() const { return resource_id_; }

  void EnqueueLoadData(int batch_size) {
    ReportCUDAErrors(
        cudaMemcpyAsync(buffers[input_index], input_buffer_,
                        batch_size * single_input_size * sizeof(float),
                        cudaMemcpyKind::cudaMemcpyHostToDevice, stream_));
  }

  void EnqueueUnloadResults(int batch_size) {
    ReportCUDAErrors(
        cudaMemcpyAsync(output_policy_buffer_, buffers[output_policy_index],
                        batch_size * single_output_pol_size * sizeof(float),
                        cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_));

    ReportCUDAErrors(
        cudaMemcpyAsync(output_value_buffer_, buffers[output_value_index],
                        batch_size * sizeof(float),
                        cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_));
  }

  void EnqueueContextExecute(int batch_size) {
    context_->enqueue(batch_size, buffers, stream_, nullptr);
  }

  void SynchronizeStream() { ReportCUDAErrors(cudaStreamSynchronize(stream_)); }

  void ExecuteComputation(int batch_size) {
    std::string range_name = "NetworkEvaluation" + std::to_string(resource_id_);
    auto range_id = nvtxRangeStartA(range_name.c_str());

    EnqueueLoadData(batch_size);
    EnqueueContextExecute(batch_size);
    EnqueueUnloadResults(batch_size);
    SynchronizeStream();

    nvtxRangeEnd(range_id);
  }
};

class ComputationResourceCache {
  nvi::ICudaEngine* engine_;
  int total_resources;

  std::mutex mutex;
  std::queue<std::unique_ptr<TensorRTComputationResource>> resources;

 public:
  ComputationResourceCache(nvi::ICudaEngine* engine)
      : engine_(engine), total_resources(0), mutex(), resources() {}

  std::unique_ptr<TensorRTComputationResource> GetNewResource() {
    std::lock_guard<std::mutex> lock(mutex);

    if (resources.empty()) {
      total_resources += 1;
      return std::make_unique<TensorRTComputationResource>(engine_,
                                                           total_resources);
    }

    std::unique_ptr<TensorRTComputationResource> result =
        std::move(resources.front());
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
  nvi::IRuntime* runtime_;
  nvi::IBuilder* builder_;
  nvi::ICudaEngine* engine_;
  nvi::IExecutionContext* context_;
  int max_batch_size_;
  std::unique_ptr<ComputationResourceCache> resource_cache_;

 public:
  TensorRTNetwork(const WeightsFile& weights, const OptionsDict options);
  virtual ~TensorRTNetwork();

  std::unique_ptr<NetworkComputation> NewComputation() override;
};

void save_serialized_model(nvi::IHostMemory* model) {
  auto file = fopen("serialized_model.dat", "wb");
  size_t size = model->size();

  fwrite(&size, sizeof(size), 1, file);
  fwrite(model->data(), 1, size, file);

  fclose(file);
}

std::tuple<nvi::ICudaEngine*, nvi::IRuntime*> load_serialized_model(
    nvi::ILogger& logger) {
  auto file = fopen("serialized_model.dat", "rb");

  if (!file) {
    return {nullptr, nullptr};
  }

  size_t size;
  fread(&size, sizeof(size), 1, file);

  char* buffer = new char[size];

  fread(buffer, 1, size, file);
  fclose(file);

  nvi::IRuntime* runtime = nvi::createInferRuntime(logger);
  nvi::ICudaEngine* engine =
      runtime->deserializeCudaEngine(buffer, size, nullptr);

  delete[] buffer;

  return std::make_tuple(engine, runtime);
}

TensorRTNetwork::TensorRTNetwork(const WeightsFile& file,
                                 const OptionsDict options)
    : context_(nullptr),
      runtime_(nullptr),
      builder_(nullptr),
      engine_(nullptr),
      max_batch_size_(0) {
  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

  std::tie(engine_, runtime_) = load_serialized_model(gLogger);

  if (!engine_) {
    LegacyWeights weights(file.weights());
    std::tie(engine_, builder_) =
        create_model(gLogger, weights, max_batch_size_);

    // serialize created model
    nvi::IHostMemory* serializedModel = engine_->serialize();
    save_serialized_model(serializedModel);
    serializedModel->destroy();
  }

  resource_cache_ = std::make_unique<ComputationResourceCache>(engine_);
}

TensorRTNetwork::~TensorRTNetwork() {
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (builder_) builder_->destroy();
  if (runtime_) runtime_->destroy();
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
  TensorRTNetworkComputation(nvi::IExecutionContext* context,
                             ComputationResourceCache* resource_cache);
  virtual ~TensorRTNetworkComputation();

  void AddInput(InputPlanes&& input) override;
  // Do the computation.
  void ComputeBlocking() override;
  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return in_batch; }
  // Returns Q value of @sample.
  float GetQVal(int sample) const override {
    assert(computation_done);
    return resource->output_value_buffer()[sample];
  }
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    assert(computation_done);
    return resource
        ->output_policy_buffer()[sample * single_output_pol_size + move_id];
  }
};

TensorRTNetworkComputation::TensorRTNetworkComputation(
    nvi::IExecutionContext* context, ComputationResourceCache* resource_cache)
    : context(context),
      cache(resource_cache),
      in_batch(0),
      computation_done(false),
      resource(resource_cache->GetNewResource()) {}

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
  resource->ExecuteComputation(in_batch);
  in_batch = 0;
  computation_done = true;
}

TensorRTNetworkComputation::~TensorRTNetworkComputation() {
  cache->ReleaseResource(std::move(resource));
}

std::unique_ptr<NetworkComputation> TensorRTNetwork::NewComputation() {
  return std::make_unique<TensorRTNetworkComputation>(context_,
                                                      resource_cache_.get());
}

std::unique_ptr<Network> MakeTensorRTNetwork(const WeightsFile& weights,
                                             const OptionsDict& options) {
  return std::make_unique<TensorRTNetwork>(weights, options);
}

REGISTER_NETWORK("tensorrt", MakeTensorRTNetwork, 100);

}  // namespace
}  // namespace tensorrt_backend
}  // namespace lczero