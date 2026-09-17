#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace robot_attention_perception
{

struct TranscriptRecord
{
  std::string track_id;
  std::string text;
  bool is_final{false};
  std::string language{"unknown"};
  float clarity{0.0F};
  std::optional<std::int64_t> started_ms;
  std::optional<std::int64_t> ended_ms;
  std::optional<std::int64_t> emitted_ms;
};

struct AcousticTrack
{
  std::string track_id;
  std::int64_t stamp_ms{0};
  bool voice_activity{false};
  float speech_probability{0.0F};
  float clarity{0.0F};
  std::optional<float> azimuth_deg;
  std::optional<float> elevation_deg;
  std::optional<float> distance_m;
  float overlap_probability{0.0F};
  float self_echo_probability{0.0F};
  std::string speaker_label{"unknown"};
  std::optional<TranscriptRecord> transcript;

  [[nodiscard]] AcousticTrack aged(std::int64_t age_ms, std::int64_t max_age_ms) const;
};

struct VisionPerson
{
  std::string person_id;
  std::int64_t stamp_ms{0};
  std::string face_id;
  std::string body_id;
  std::string voice_id;
  std::string role{"unknown"};
  std::optional<float> azimuth_deg;
  std::optional<float> elevation_deg;
  std::optional<float> distance_m;
  float distance_confidence{0.0F};
  std::string depth_source{"none"};
  std::string proxemic_space{"unknown"};
  std::string engagement_status{"unknown"};
  bool face_visible{false};
  float face_confidence{0.0F};
  float mouth_open_ratio{0.0F};
  bool lip_motion{false};
  std::vector<float> mouth_roi_features;
  float gaze_score{0.0F};
  float body_facing_score{0.0F};
  float bbox_area_ratio{0.0F};
  std::string gesture;
  float gesture_score{0.0F};
  float identity_confidence{0.0F};
  float emotion_valence{0.0F};
  float emotion_arousal{0.0F};
  bool emotion_valid{false};
  std::string emotion_label{"unknown"};

  [[nodiscard]] VisionPerson aged(std::int64_t age_ms, std::int64_t max_age_ms) const;
};

struct RobotState
{
  std::int64_t stamp_ms{0};
  bool speaking{false};
  bool moving{false};
  float playback_rms_db{0.0F};
};

struct PerceptionFrame
{
  std::int64_t stamp_ms{0};
  std::vector<AcousticTrack> acoustic_tracks;
  std::vector<VisionPerson> vision_people;
  RobotState robot;
};

struct AttentionState
{
  std::int64_t stamp_ms{0};
  std::string target_id{"none"};
  std::string target_kind{"none"};
  float confidence{0.0F};
  bool listen{false};
  bool addressed_to_robot{false};
  std::optional<std::string> source_track_id;
  std::optional<std::string> person_id;
  std::optional<float> azimuth_deg;
  std::vector<std::string> reasons;
  std::vector<TranscriptRecord> transcripts;
};

float freshnessFactor(std::int64_t age_ms, std::int64_t max_age_ms);

}  // namespace robot_attention_perception
