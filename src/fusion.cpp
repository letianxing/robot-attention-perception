#include "robot_attention_perception/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace robot_attention_perception
{
namespace
{

struct Candidate
{
  float score{0.0F};
  AcousticTrack acoustic;
  std::optional<VisionPerson> vision;
  std::vector<std::string> reasons;
};

std::unordered_map<std::string, AttentionFusion::Factory> & registry()
{
  static std::unordered_map<std::string, AttentionFusion::Factory> map;
  return map;
}

std::string targetId(const Candidate & candidate)
{
  if (candidate.vision.has_value()) {
    return "person:" + candidate.vision->person_id;
  }
  return "sound:" + candidate.acoustic.track_id;
}

std::optional<float> mergedAzimuth(const Candidate & candidate)
{
  if (candidate.vision.has_value() && candidate.vision->azimuth_deg.has_value()) {
    return candidate.vision->azimuth_deg;
  }
  return candidate.acoustic.azimuth_deg;
}

bool isConversationalGesture(const std::string & gesture)
{
  return gesture == "wave" || gesture == "invite" || gesture == "call";
}

float commonSourceProbability(
  const AcousticTrack & track,
  const VisionPerson & person,
  const FusionConfig & config)
{
  const auto age_ms = std::abs(track.stamp_ms - person.stamp_ms);
  const float temporal_sigma = std::max(1.0F, config.causal_temporal_sigma_ms);
  const float temporal = std::exp(
    -0.5F * std::pow(static_cast<float>(age_ms) / temporal_sigma, 2.0F));
  float spatial = 0.72F;
  if (track.azimuth_deg.has_value() && person.azimuth_deg.has_value()) {
    const float sigma = std::max(1.0F, config.causal_spatial_sigma_deg);
    const float delta = angularDistanceDeg(*track.azimuth_deg, *person.azimuth_deg);
    spatial = std::exp(-0.5F * std::pow(delta / sigma, 2.0F));
  }
  const float audio_reliability = 0.5F * clamp(track.speech_probability) +
    0.5F * clamp(track.clarity);
  const float visual_reliability = (
    clamp(person.face_confidence) + clamp(person.gaze_score) +
    clamp(person.body_facing_score)) / 3.0F;
  const float reliability = std::max(0.05F, audio_reliability) *
    std::max(0.05F, visual_reliability);
  const float common_likelihood = spatial * temporal * (0.35F + 0.65F * reliability);
  const float prior = clamp(config.common_source_prior, 0.01F, 0.99F);
  const float numerator = prior * common_likelihood;
  return numerator / std::max(1e-9F, numerator + ((1.0F - prior) * 0.22F));
}

class WeightedAudioVisualAttention : public AttentionAlgorithm
{
public:
  std::string name() const override { return "weighted_audio_visual"; }
  std::string description() const override
  {
    return "weighted VAD/DOA/clarity/gaze/facing/identity attention score";
  }

  AttentionState update(
    const PerceptionFrame & frame,
    const FusionConfig & config,
    AttentionMemory & memory) override
  {
    std::vector<Candidate> candidates;
    for (const auto & track : frame.acoustic_tracks) {
      auto candidate = scoreTrack(frame, track, config, memory);
      if (candidate.score >= config.min_attention_score) {
        candidates.push_back(candidate);
      }
    }

    std::vector<TranscriptRecord> transcripts;
    for (const auto & track : frame.acoustic_tracks) {
      if (track.transcript.has_value()) {
        transcripts.push_back(*track.transcript);
      }
    }

    if (candidates.empty()) {
      memory.last_target_id.clear();
      AttentionState state;
      state.stamp_ms = frame.stamp_ms;
      state.reasons = {"no_active_candidate"};
      state.transcripts = transcripts;
      return state;
    }

    auto best = std::max_element(
      candidates.begin(), candidates.end(),
      [](const Candidate & lhs, const Candidate & rhs) { return lhs.score < rhs.score; });
    memory.last_target_id = targetId(*best);

    AttentionState state;
    state.stamp_ms = frame.stamp_ms;
    state.target_id = memory.last_target_id;
    state.target_kind = best->vision.has_value() ? "person" : "sound_source";
    state.confidence = clamp(best->score);
    state.listen = best->score >= config.min_listen_score;
    state.addressed_to_robot = isAddressed(*best, config);
    state.source_track_id = best->acoustic.track_id;
    if (best->vision.has_value()) {
      state.person_id = best->vision->person_id;
    }
    state.azimuth_deg = mergedAzimuth(*best);
    state.reasons = best->reasons;
    state.transcripts = transcripts;
    return state;
  }

protected:
  virtual std::pair<std::optional<VisionPerson>, std::vector<std::string>> associate(
    const AcousticTrack & track,
    const std::vector<VisionPerson> & people,
    const FusionConfig & config)
  {
    if (people.empty()) {
      return {std::nullopt, {}};
    }
    if (!track.azimuth_deg.has_value()) {
      auto visible_count = std::count_if(
        people.begin(), people.end(),
        [](const VisionPerson & person) { return person.face_visible; });
      if (visible_count == 1) {
        auto it = std::find_if(
          people.begin(), people.end(),
          [](const VisionPerson & person) { return person.face_visible; });
        const float probability = commonSourceProbability(track, *it, config);
        if (probability >= config.causal_association_threshold) {
          return {*it, {
            "matched_single_visible:" + it->person_id,
            "causal_common:" + std::to_string(probability)}};
        }
      }
      return {std::nullopt, {"causal_separate:ambiguous_mono_audio"}};
    }

    const VisionPerson * best = nullptr;
    float best_delta = 999.0F;
    float best_probability = 0.0F;
    for (const auto & person : people) {
      if (!person.azimuth_deg.has_value()) {
        continue;
      }
      const float delta = angularDistanceDeg(*track.azimuth_deg, *person.azimuth_deg);
      const float probability = commonSourceProbability(track, person, config);
      if (delta <= config.association_max_angle_deg &&
        probability >= config.causal_association_threshold && probability > best_probability)
      {
        best = &person;
        best_delta = delta;
        best_probability = probability;
      }
    }
    if (best == nullptr) {
      return {std::nullopt, {"causal_separate:spatiotemporal_conflict"}};
    }
    return {*best, {
      "matched_bearing:" + best->person_id,
      "causal_common:" + std::to_string(best_probability)}};
  }

  virtual float scoreVision(
    const VisionPerson & vision,
    const FusionConfig & config,
    std::vector<std::string> & reasons)
  {
    float score = 0.0F;
    score += config.gaze_weight * clamp(vision.gaze_score);
    score += config.facing_weight * clamp(vision.body_facing_score);
    if (vision.distance_m.has_value() && vision.distance_confidence > 0.0F) {
      const float distance_score = clamp((4.5F - *vision.distance_m) / 4.0F);
      score += config.distance_weight * distance_score * clamp(vision.distance_confidence);
      reasons.push_back("depth:" + vision.depth_source);
    } else {
      score += config.distance_weight * clamp(vision.bbox_area_ratio * 4.0F);
    }
    reasons.push_back("matched_vision:" + vision.person_id);
    if (vision.gaze_score > 0.55F) {
      reasons.push_back("gaze_to_robot");
    }
    if (vision.body_facing_score > 0.55F) {
      reasons.push_back("body_facing_robot");
    }
    if (vision.role == "owner" || vision.role == "known") {
      score += config.identity_weight * std::max(0.5F, clamp(vision.identity_confidence));
      reasons.push_back("role:" + vision.role);
    }
    if (isConversationalGesture(vision.gesture)) {
      score += config.gesture_weight;
      reasons.push_back("gesture:" + vision.gesture);
    }
    return score;
  }

  Candidate scoreTrack(
    const PerceptionFrame & frame,
    const AcousticTrack & track,
    const FusionConfig & config,
    AttentionMemory & memory)
  {
    Candidate candidate;
    candidate.acoustic = track;
    if (!track.voice_activity) {
      candidate.reasons = {"no_voice_activity"};
      return candidate;
    }
    if (frame.robot.speaking &&
      track.self_echo_probability >= config.self_echo_reject_threshold)
    {
      candidate.reasons = {"rejected_self_echo", "robot_playback_active"};
      return candidate;
    }

    auto [vision, association_reasons] = associate(track, frame.vision_people, config);
    candidate.vision = vision;
    candidate.reasons = association_reasons;

    candidate.score += config.audio_weight * clamp(track.speech_probability);
    candidate.score += config.clarity_weight * clamp(track.clarity);
    if (track.speech_probability > 0.5F) {
      candidate.reasons.push_back("speech");
    }
    if (track.clarity > 0.65F) {
      candidate.reasons.push_back("clear_voice");
    }

    if (candidate.vision.has_value()) {
      candidate.score += scoreVision(*candidate.vision, config, candidate.reasons);
    }

    if (config.use_transcript_wake_words && hasWakeWord(track.transcript)) {
      candidate.score += 0.12F;
      candidate.reasons.push_back("wake_word");
    }

    if (track.overlap_probability > 0.4F) {
      candidate.score -= config.overlap_penalty * clamp(track.overlap_probability);
      candidate.reasons.push_back("overlap");
    }
    if (track.self_echo_probability > 0.2F) {
      candidate.score -= config.echo_penalty * clamp(track.self_echo_probability);
      candidate.reasons.push_back("self_echo");
    }
    if (frame.robot.speaking && track.self_echo_probability > 0.1F) {
      candidate.score -= config.robot_speaking_echo_penalty;
      candidate.reasons.push_back("robot_playback_active");
    }
    if (!memory.last_target_id.empty() && targetId(candidate) == memory.last_target_id) {
      candidate.score += config.hysteresis_bonus;
      candidate.reasons.push_back("hysteresis");
    }
    candidate.score = std::max(0.0F, candidate.score);
    return candidate;
  }

  virtual bool isAddressed(const Candidate & candidate, const FusionConfig & config)
  {
    if (candidate.score < config.min_listen_score) {
      return false;
    }
    if (config.use_transcript_wake_words && hasWakeWord(candidate.acoustic.transcript)) {
      return true;
    }
    if (!candidate.vision.has_value()) {
      return false;
    }
    return candidate.vision->gaze_score >= 0.58F ||
      candidate.vision->body_facing_score >= 0.68F ||
      isConversationalGesture(candidate.vision->gesture);
  }
};

class Ros4HriNativeAttention final : public WeightedAudioVisualAttention
{
public:
  std::string name() const override { return "ros4hri_native"; }
  std::string description() const override
  {
    return "ROS4HRI-style person/voice association with engagement and proxemics";
  }

protected:
  std::pair<std::optional<VisionPerson>, std::vector<std::string>> associate(
    const AcousticTrack & track,
    const std::vector<VisionPerson> & people,
    const FusionConfig & config) override
  {
    for (const auto & person : people) {
      if (!person.voice_id.empty() && person.voice_id == track.track_id) {
        return {person, {"ros4hri_voice_match:" + person.person_id}};
      }
    }
    return WeightedAudioVisualAttention::associate(track, people, config);
  }

  float scoreVision(
    const VisionPerson & vision,
    const FusionConfig & config,
    std::vector<std::string> & reasons) override
  {
    float score = WeightedAudioVisualAttention::scoreVision(vision, config, reasons);
    if (vision.engagement_status == "engaged" || vision.engagement_status == "engaging") {
      score += 0.10F;
      reasons.push_back("engagement:" + vision.engagement_status);
    }
    if (vision.proxemic_space == "intimate" || vision.proxemic_space == "personal" ||
      vision.proxemic_space == "social")
    {
      score += 0.04F;
      reasons.push_back("proxemic:" + vision.proxemic_space);
    }
    return score;
  }

  bool isAddressed(const Candidate & candidate, const FusionConfig & config) override
  {
    if (candidate.vision.has_value() && candidate.vision->engagement_status == "engaged") {
      return candidate.score >= config.min_listen_score;
    }
    return WeightedAudioVisualAttention::isAddressed(candidate, config);
  }
};

struct RegistryInitializer
{
  RegistryInitializer()
  {
    AttentionFusion::registerAlgorithm(
      "ros4hri_native",
      []() { return std::make_unique<Ros4HriNativeAttention>(); });
    AttentionFusion::registerAlgorithm(
      "weighted_audio_visual",
      []() { return std::make_unique<WeightedAudioVisualAttention>(); });
  }
};

const RegistryInitializer registry_initializer;

}  // namespace

AttentionFusion::AttentionFusion(FusionConfig config)
: config_(std::move(config)), algorithm_(createAlgorithm(config_.algorithm))
{
}

AttentionState AttentionFusion::update(const PerceptionFrame & frame)
{
  return algorithm_->update(frame, config_, memory_);
}

void AttentionFusion::registerAlgorithm(const std::string & name, Factory factory)
{
  registry()[name] = std::move(factory);
}

std::vector<std::string> AttentionFusion::availableAlgorithms()
{
  std::vector<std::string> names;
  for (const auto & item : registry()) {
    names.push_back(item.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::unique_ptr<AttentionAlgorithm> AttentionFusion::createAlgorithm(const std::string & name)
{
  const auto it = registry().find(name);
  if (it == registry().end()) {
    throw std::invalid_argument("unknown attention algorithm: " + name);
  }
  return it->second();
}

float clamp(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

float angularDistanceDeg(float a, float b)
{
  float delta = std::fmod(a - b + 180.0F, 360.0F);
  if (delta < 0.0F) {
    delta += 360.0F;
  }
  delta = std::abs(delta - 180.0F);
  return std::min(delta, 360.0F - delta);
}

bool hasWakeWord(const std::optional<TranscriptRecord> & transcript)
{
  if (!transcript.has_value()) {
    return false;
  }
  std::string text = transcript->text;
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  const std::vector<std::string> wake_words{
    "reachy", "hey reachy", "机器人", "小睿", "小瑞", "小锐"};
  return std::any_of(
    wake_words.begin(), wake_words.end(),
    [&text](const std::string & word) { return text.find(word) != std::string::npos; });
}

}  // namespace robot_attention_perception
