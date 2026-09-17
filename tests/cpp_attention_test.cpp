#include "robot_attention_perception/async_fusion.hpp"
#include "robot_attention_perception/fusion.hpp"

#include <cstdlib>
#include <iostream>

namespace
{

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main()
{
  using namespace robot_attention_perception;
  const auto algorithms = AttentionFusion::availableAlgorithms();
  require(!algorithms.empty(), "algorithms should be registered");
  require(
    std::find(algorithms.begin(), algorithms.end(), "ros4hri_native") != algorithms.end(),
    "default ros4hri_native algorithm should exist");

  PerceptionFrame frame;
  frame.stamp_ms = 1000;
  frame.acoustic_tracks.push_back(
    AcousticTrack{
      "voice_a", 1000, true, 0.88F, 0.82F, 5.0F, std::nullopt, std::nullopt,
      0.2F, 0.01F});
  frame.acoustic_tracks.push_back(
    AcousticTrack{
      "voice_b", 1000, true, 0.86F, 0.76F, -42.0F, std::nullopt, std::nullopt,
      0.55F, 0.01F});

  VisionPerson owner;
  owner.person_id = "owner";
  owner.stamp_ms = 1000;
  owner.voice_id = "voice_a";
  owner.role = "owner";
  owner.azimuth_deg = 8.0F;
  owner.proxemic_space = "social";
  owner.engagement_status = "engaged";
  owner.face_visible = true;
  owner.gaze_score = 0.82F;
  owner.body_facing_score = 0.78F;
  owner.bbox_area_ratio = 0.18F;
  owner.identity_confidence = 0.91F;
  frame.vision_people.push_back(owner);

  VisionPerson side;
  side.person_id = "side_guest";
  side.stamp_ms = 1000;
  side.voice_id = "voice_b";
  side.role = "known";
  side.azimuth_deg = -40.0F;
  side.face_visible = true;
  side.gaze_score = 0.2F;
  side.body_facing_score = 0.3F;
  frame.vision_people.push_back(side);

  AttentionFusion fusion;
  const auto state = fusion.update(frame);
  require(state.person_id.has_value() && *state.person_id == "owner", "attention should select engaged owner");
  require(state.source_track_id.has_value() && *state.source_track_id == "voice_a", "attention should select matched voice");
  require(state.listen, "selected target should be listenable");
  require(state.addressed_to_robot, "engaged person should be addressed to robot");

  frame.robot.speaking = true;
  frame.acoustic_tracks.clear();
  frame.acoustic_tracks.push_back(
    AcousticTrack{
      "robot_echo", 1200, true, 0.96F, 0.9F, 0.0F, std::nullopt, std::nullopt,
      0.1F, 0.95F});
  const auto echo_state = fusion.update(frame);
  require(echo_state.target_id == "none", "robot echo should not become attention target");

  AsyncPerceptionBuffer buffer;
  buffer.ingestVision({owner});
  buffer.ingestAcoustic({
    AcousticTrack{
      "voice_a", 1120, true, 0.88F, 0.82F, 6.0F, std::nullopt, std::nullopt,
      0.2F, 0.01F}});
  const auto async_state = buffer.update(1130);
  require(async_state.has_value(), "async buffer should produce a state");
  require(
    async_state->person_id.has_value() && *async_state->person_id == "owner",
    "async buffer should fuse recent visual context with fresh audio");
  const auto rate_limited = buffer.update(1135);
  require(!rate_limited.has_value(), "async buffer should rate-limit excessive updates");

  return 0;
}
