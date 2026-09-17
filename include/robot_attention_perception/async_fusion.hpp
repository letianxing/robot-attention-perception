#pragma once

#include "robot_attention_perception/fusion.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace robot_attention_perception
{

struct AsyncFusionConfig
{
  std::int64_t audio_ttl_ms{180};
  std::int64_t vision_ttl_ms{450};
  std::int64_t robot_ttl_ms{250};
  float max_frame_rate_hz{50.0F};
};

class AsyncPerceptionBuffer
{
public:
  explicit AsyncPerceptionBuffer(
    FusionConfig fusion_config = {}, AsyncFusionConfig async_config = {});

  void ingestAcoustic(const std::vector<AcousticTrack> & tracks);
  void ingestVision(const std::vector<VisionPerson> & people);
  void ingestRobot(const RobotState & state);

  std::optional<AttentionState> update(std::int64_t now_ms);

private:
  void prune(std::int64_t now_ms);
  static void appendReason(AttentionState & state, const std::string & reason);

  AttentionFusion fusion_;
  AsyncFusionConfig config_;
  std::unordered_map<std::string, AcousticTrack> acoustic_tracks_;
  std::unordered_map<std::string, VisionPerson> vision_people_;
  RobotState robot_state_;
  std::int64_t last_update_ms_{0};
};

}  // namespace robot_attention_perception
