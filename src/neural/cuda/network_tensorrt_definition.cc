#include "network_tensorrt_definition.h"
#include "cuda_common.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

#include "NvInfer.h"

namespace nvi = nvinfer1;

namespace lczero {
namespace {

nvi::Weights vector_to_weights(FloatVector const& vector) {
  return {nvi::DataType::kFLOAT, vector.data(),
          static_cast<int64_t>(vector.size())};
}

nvi::ITensor* name_and_get_output(nvi::ILayer* layer, std::string name) {
  layer->setName(name.c_str());
  return layer->getOutput(0);
}

bool is_identity_elementwise(const FloatVector& values, float identity = 0) {
  return std::all_of(values.begin(), values.end(),
                     [=](float x) { return x == identity; });
}

nvi::ITensor* process_bn(const FloatVector& bn_means,
                         const FloatVector& bn_stddivs,
                         const FloatVector& bn_gammas,
                         const FloatVector& bn_betas, nvi::ITensor* input,
                         nvi::INetworkDefinition* network,
                         std::string name = "") {
  auto num_channels = input->getDimensions().d[0];
  nvi::Dims3 dims_channel{num_channels, 1, 1};
  nvi::DataType dt = nvi::DataType::kFLOAT;

  if (!is_identity_elementwise(bn_means, 0.0)) {
    auto bn_means_c = name_and_get_output(
        network->addConstant(dims_channel, vector_to_weights(bn_means)),
        name + "/constant_mean");
    input = name_and_get_output(
        network->addElementWise(*input, *bn_means_c,
                                nvi::ElementWiseOperation::kMIN),
        name + "/mean_sub");
  }

  if (!is_identity_elementwise(bn_stddivs, 1.0)) {
    auto bn_stddivs_c = name_and_get_output(
        network->addConstant(dims_channel, vector_to_weights(bn_stddivs)),
        name + "/constant_stddiv");
    input = name_and_get_output(
        network->addElementWise(*input, *bn_stddivs_c,
                                nvi::ElementWiseOperation::kDIV),
        name + "/std_div");
  }

  if (!is_identity_elementwise(bn_betas, 0.0)) {
    auto bn_betas_c =
        network->addConstant(dims_channel, vector_to_weights(bn_betas))
            ->getOutput(0);
    input = network
                ->addElementWise(*input, *bn_betas_c,
                                 nvi::ElementWiseOperation::kSUM)
                ->getOutput(0);
  }

  if (!is_identity_elementwise(bn_gammas, 1.0)) {
    auto bn_gammas_c =
        network->addConstant(dims_channel, vector_to_weights(bn_gammas))
            ->getOutput(0);
    input = network
                ->addElementWise(*input, *bn_gammas_c,
                                 nvi::ElementWiseOperation::kPROD)
                ->getOutput(0);
  }

  return input;
}

void simplify_conv_block(LegacyWeights::ConvBlock& block,
                         bool foldBNLayer = false) {
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
                                int numOutputMaps, nvi::ITensor* input,
                                nvi::INetworkDefinition* network, int size = 3,
                                std::string name = "") {
  if (numOutputMaps == -1) {
    numOutputMaps = conv_desc.bn_means.size();
  }

  simplify_conv_block(conv_desc, true);

  auto conv = network->addConvolution(*input, numOutputMaps, {size, size},
                                      vector_to_weights(conv_desc.weights),
                                      vector_to_weights(conv_desc.biases));

  conv->setPadding({(size - 1) / 2, (size - 1) / 2});
  conv->setName(name.c_str());

  return process_bn(conv_desc.bn_means, conv_desc.bn_stddivs,
                    conv_desc.bn_gammas, conv_desc.bn_betas, conv->getOutput(0),
                    network, name + "/bn");
}

nvi::ITensor* process_residual(lczero::LegacyWeights::Residual& desc,
                               nvi::ITensor* input,
                               nvi::INetworkDefinition* network,
                               std::string name = "") {
  int num_filters = input->getDimensions().d[0];
  auto current = input;

  current = process_convblock(desc.conv1, num_filters, current, network, 3,
                              name + "/conv1");
  current = name_and_get_output(
      network->addActivation(*current, nvi::ActivationType::kRELU),
      name + "/activation1");
  current = process_convblock(desc.conv2, num_filters, current, network, 3,
                              name + "/conv2");
  current = name_and_get_output(
      network->addElementWise(*input, *current,
                              nvi::ElementWiseOperation::kSUM),
      name + "/residual");
  current = name_and_get_output(
      network->addActivation(*current, nvi::ActivationType::kRELU),
      name + "/activation2");

  return current;
}

nvi::ITensor* process_policy_head(LegacyWeights::ConvBlock& conv,
                                  const FloatVector& ip_pol_w,
                                  const FloatVector& ip_pol_b,
                                  nvi::ITensor* input,
                                  nvi::INetworkDefinition* network) {
  auto current = input;
  current = process_convblock(conv, 32, current, network, 1, "policy/conv");
  current = name_and_get_output(
      network->addFullyConnected(*current, ip_pol_b.size(),
                                 vector_to_weights(ip_pol_w),
                                 vector_to_weights(ip_pol_b)),
      "policy/fc");

  current =
      name_and_get_output(network->addSoftMax(*current), "policy/softmax");

  return current;
}

nvi::ITensor* process_value_head(LegacyWeights::ConvBlock& conv,
                                 const FloatVector& ip1_val_w,
                                 const FloatVector& ip1_val_b,
                                 const FloatVector& ip2_val_w,
                                 const FloatVector& ip2_val_b,
                                 nvi::ITensor* input,
                                 nvi::INetworkDefinition* network) {
  auto current = input;
  current = process_convblock(conv, 32, current, network, 1, "value/conv");
  current = name_and_get_output(
      network->addFullyConnected(*current, 128, vector_to_weights(ip1_val_w),
                                 vector_to_weights(ip1_val_b)),
      "value/fc1");
  current = name_and_get_output(
      network->addActivation(*current, nvi::ActivationType::kRELU),
      "value/activation1");
  current = name_and_get_output(
      network->addFullyConnected(*current, 1, vector_to_weights(ip2_val_w),
                                 vector_to_weights(ip2_val_b)),
      "value/fc2");
  current = name_and_get_output(
      network->addActivation(*current, nvi::ActivationType::kTANH),
      "value/output");

  return current;
}

}  // namespace

namespace tensorrt_backend {

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_POL_BLOB_NAME = "output_policy";
const char* OUTPUT_VAL_BLOB_NAME = "output_value";

std::tuple<nvi::ICudaEngine*, nvi::IBuilder*> create_model(
    nvi::ILogger& logger, LegacyWeights& weights, int max_batch_size) {
  nvi::IBuilder* builder = nvi::createInferBuilder(logger);
  nvi::INetworkDefinition* network = builder->createNetwork();

  builder->setFp16Mode(true);

  nvi::ITensor* current = network->addInput(
      INPUT_BLOB_NAME, nvi::DataType::kFLOAT, nvi::Dims3(112, 8, 8));
  current = process_convblock(weights.input, 256, current, network, 3,
                              "initial_conv");

  // compute residual tower
  int counter = 1;
  for (auto&& block : weights.residual) {
    current = process_residual(block, current, network,
                               "residual_block/" + std::to_string(counter));
    counter += 1;
  }

  // policy head
  auto output_pol = process_policy_head(weights.policy, weights.ip_pol_w,
                                        weights.ip_pol_b, current, network);
  output_pol->setName(OUTPUT_POL_BLOB_NAME);

  // value head
  auto output_val = process_value_head(weights.value, weights.ip1_val_w,
                                       weights.ip1_val_b, weights.ip2_val_w,
                                       weights.ip2_val_b, current, network);
  output_val->setName(OUTPUT_VAL_BLOB_NAME);

  network->markOutput(*output_pol);
  network->markOutput(*output_val);

  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(1 << 25);

  nvi::ICudaEngine* engine = builder->buildCudaEngine(*network);
  network->destroy();

  return std::make_tuple(engine, builder);
}

}  // namespace tensorrt_backend
}  // namespace lczero
