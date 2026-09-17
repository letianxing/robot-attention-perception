#pragma once

#include "robot_attention_perception/types.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace robot_attention_perception
{

struct FusionConfig
{
  std::string algorithm{"ros4hri_native"};
  float association_max_angle_deg{35.0F};
  float min_attention_score{0.52F};
  float min_listen_score{0.62F};
  float hysteresis_bonus{0.10F};
  bool use_transcript_wake_words{true};
  float audio_weight{0.45F};
  float clarity_weight{0.20F};
  float gaze_weight{0.18F};
  float facing_weight{0.14F};
  float identity_weight{0.08F};
  float gesture_weight{0.08F};
  float distance_weight{0.05F};
  float overlap_penalty{0.12F};
  float echo_penalty{0.45F};
  float robot_speaking_echo_penalty{0.18F};
  float self_echo_reject_threshold{0.75F};
  float common_source_prior{0.62F};
  float causal_association_threshold{0.48F};
  float causal_spatial_sigma_deg{18.0F};
  float causal_temporal_sigma_ms{120.0F};
};

struct AttentionMemory
{
  std::string last_target_id;
};

class AttentionAlgorithm
{
public:
  virtual ~AttentionAlgorithm() = default;
  virtual std::string name() const = 0;
  virtual std::string description() const = 0;
  virtual AttentionState update(
    const PerceptionFrame & frame,
    const FusionConfig & config,
    AttentionMemory & memory) = 0;
};

class AttentionFusion
{
public:
  using Factory = std::function<std::unique_ptr<AttentionAlgorithm>()>;

  explicit AttentionFusion(FusionConfig config = {});

  AttentionState update(const PerceptionFrame & frame);

  static void registerAlgorithm(const std::string & name, Factory factory);
  static std::vector<std::string> availableAlgorithms();
  static std::unique_ptr<AttentionAlgorithm> createAlgorithm(const std::string & name);

private:
  FusionConfig config_;
  AttentionMemory memory_;
  std::unique_ptr<AttentionAlgorithm> algorithm_;
};

float clamp(float value, float low = 0.0F, float high = 1.0F);
float angularDistanceDeg(float a, float b);
bool hasWakeWord(const std::optional<TranscriptRecord> & transcript);

}  // namespace robot_attention_perception
