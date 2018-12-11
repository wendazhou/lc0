#pragma once

#include "NvInfer.h"
#include <tuple>
#include "neural/network_legacy.h"

namespace lczero {
namespace tensorrt_backend {


// TensorRT backend shared information

// TensorRT uses strings to refer to the input and output buffers.
// Here we define shared values for the input, policy output and value output buffers.
extern const char* INPUT_BLOB_NAME;
extern const char* OUTPUT_POL_BLOB_NAME;
extern const char* OUTPUT_VAL_BLOB_NAME;

// Creates a model from the given weights and operating at the given batch size.
// Ownership of the created Builder and Engine is transferred to the caller.
std::tuple<nvinfer1::ICudaEngine*, nvinfer1::IBuilder*> create_model(nvinfer1::ILogger& logger, LegacyWeights& weights, int max_batch_size);

}
}  // namespace lczero