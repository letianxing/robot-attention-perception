#include "robot_attention_perception/async_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace robot_attention_perception
{

float freshnessFactor(std::int64_t age_ms, std::int64_t max_age_ms)
{
  if (max_age_ms <= 0) {
    return 0.0F;
  }
  if (age_ms <= 0) {
    return 1.0F;
  }
  if (age_ms >= max_age_ms) {
    return 0.0F;
  }
  return 1.0F - (static_cast<float>(age_ms) / static_cast<float>(max_age_ms));
}

AcousticTrack AcousticTrack::aged(std::int64_t age_ms, std::int64_t max_age_ms) const
{
  if (age_ms <= 0) {
    return *this;
  }
  const float factor = freshnessFactor(age_ms, max_age_ms);
  AcousticTrack out = *this;
  out.voice_activity = out.voice_activity && factor > 0.0F;
  out.speech_probability *= factor;
  out.clarity *= factor;
  return out;
}

VisionPerson VisionPerson::aged(std::int64_t age_ms, std::int64_t max_age_ms) const
{
  if (age_ms <= 0) {
    return *this;
  }
  const float factor = freshnessFactor(age_ms, max_age_ms);
  VisionPerson out = *this;
  out.face_visible = out.face_visible && factor > 0.0F;
  out.face_confidence *= factor;
  out.gaze_score *= factor;
  out.body_facing_score *= factor;
  out.gesture_score *= factor;
  if (factor <= 0.25F) {
    out.engagement_status = "unknown";
  }
  if (factor <= 0.45F) {
    out.gesture.clear();
  }
  return out;
}

AsyncPerceptionBuffer::AsyncPerceptionBuffer(
  FusionConfig fusion_config, AsyncFusionConfig async_config)
: fusion_(std::move(fusion_config)), config_(async_config)
{
}

void AsyncPerceptionBuffer::ingestAcoustic(const std::vector<AcousticTrack> & tracks)
{
  for (const auto & track : tracks) {
    acoustic_tracks_[track.track_id] = track;
  }
}

void AsyncPerceptionBuffer::ingestVision(const std::vector<VisionPerson> & people)
{
  for (const auto & person : people) {
    vision_people_[person.person_id] = person;
  }
}

void AsyncPerceptionBuffer::ingestRobot(const RobotState & state)
{
  robot_state_ = state;
}

std::optional<AttentionState> AsyncPerceptionBuffer::update(std::int64_t now_ms)
{
  const auto min_period_ms = static_cast<std::int64_t>(
    std::floor(1000.0F / std::max(config_.max_frame_rate_hz, 1.0F)));
  if (last_update_ms_ != 0 && now_ms - last_update_ms_ < min_period_ms) {
    return std::nullopt;
  }
  last_update_ms_ = now_ms;
  prune(now_ms);

  PerceptionFrame frame;
  frame.stamp_ms = now_ms;
  for (const auto & item : acoustic_tracks_) {
    frame.acoustic_tracks.push_back(
      item.second.aged(now_ms - item.second.stamp_ms, config_.audio_ttl_ms));
  }
  for (const auto & item : vision_people_) {
    frame.vision_people.push_back(
      item.second.aged(now_ms - item.second.stamp_ms, config_.vision_ttl_ms));
  }
  if (now_ms - robot_state_.stamp_ms <= config_.robot_ttl_ms) {
    frame.robot = robot_state_;
  } else {
    frame.robot.stamp_ms = now_ms;
  }

  auto state = fusion_.update(frame);
  if (std::any_of(
      acoustic_tracks_.begin(), acoustic_tracks_.end(),
      [now_ms](const auto & item) { return now_ms - item.second.stamp_ms > 0; }))
  {
    appendReason(state, "async_audio_cache");
  }
  if (std::any_of(
      vision_people_.begin(), vision_people_.end(),
      [now_ms](const auto & item) { return now_ms - item.second.stamp_ms > 0; }))
  {
    appendReason(state, "async_vision_cache");
  }
  return state;
}

void AsyncPerceptionBuffer::prune(std::int64_t now_ms)
{
  for (auto it = acoustic_tracks_.begin(); it != acoustic_tracks_.end();) {
    if (now_ms - it->second.stamp_ms > config_.audio_ttl_ms) {
      it = acoustic_tracks_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = vision_people_.begin(); it != vision_people_.end();) {
    if (now_ms - it->second.stamp_ms > config_.vision_ttl_ms) {
      it = vision_people_.erase(it);
    } else {
      ++it;
    }
  }
}

void AsyncPerceptionBuffer::appendReason(AttentionState & state, const std::string & reason)
{
  state.reasons.push_back(reason);
}

}  // namespace robot_attention_perception
