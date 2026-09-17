#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace attention_stack
{

constexpr std::size_t kBins = 36;

struct PersonCue
{
  std::string person_id;
  std::string voice_id;
  std::int64_t stamp_ms{0};
  std::optional<float> azimuth_deg;
  float face_confidence{0.0F};
  float gaze_score{0.0F};
  float facing_score{0.0F};
  bool lip_motion{false};
  float mouth_open_ratio{0.0F};
};

struct VoiceCue
{
  std::string track_id;
  std::int64_t stamp_ms{0};
  std::optional<float> azimuth_deg;
  bool active{false};
  float speech_probability{0.0F};
  float clarity{0.0F};
  float overlap_probability{0.0F};
  float self_echo_probability{0.0F};
  float target_speaker_probability{1.0F};
  bool tse_enabled{false};
  bool tse_healthy{true};
  float tse_latency_ms{0.0F};
  bool target_speech_rejected{false};
};

enum class AttentionPhase {IDLE, ORIENTING, ENGAGED, RELEASE};

struct AttentionConfig
{
  int binding_window_ms{100};
  int debounce_ms{300};
  int working_memory_ms{8000};
  float habituation_tau_ms{5000.0F};
  float common_source_prior{0.62F};
  float engage_threshold{0.52F};
  float release_threshold{0.22F};
  float lateral_inhibition{0.18F};
  float hebbian_rate{0.025F};
};

struct AttentionOutput
{
  std::int64_t stamp_ms{0};
  std::string target_id{"none"};
  std::string target_kind{"none"};
  std::string person_id;
  std::string voice_id;
  std::optional<float> azimuth_deg;
  float confidence{0.0F};
  float common_source_probability{0.0F};
  bool listen{false};
  bool addressed_to_robot{false};
  bool fission_active{false};
  AttentionPhase phase{AttentionPhase::IDLE};
  std::vector<std::string> reasons;
  std::array<float, kBins> visual_saliency{};
  std::array<float, kBins> audio_saliency{};
  std::array<float, kBins> competition{};
  std::array<float, kBins> memory_saliency{};
};

class AttentionCore
{
public:
  explicit AttentionCore(AttentionConfig config = {});

  AttentionOutput update(
    std::int64_t stamp_ms,
    const std::vector<PersonCue> & people,
    const std::vector<VoiceCue> & voices);

private:
  AttentionConfig config_;
  AttentionPhase phase_{AttentionPhase::IDLE};
  std::int64_t phase_since_ms_{0};
  std::int64_t last_target_ms_{0};
  std::int64_t last_update_ms_{0};
  int last_bin_{-1};
  std::array<float, kBins> habituation_{};
  std::array<std::array<float, kBins>, kBins> hebbian_{};
  std::unordered_map<std::string, bool> previous_voice_activity_;

  static int azimuthBin(float azimuth_deg);
  static float binAzimuth(int bin);
  float commonSourceProbability(const PersonCue & person, const VoiceCue & voice) const;
  void updatePhase(float confidence, int winner_bin, std::int64_t stamp_ms);
};

std::string phaseName(AttentionPhase phase);

}  // namespace attention_stack
