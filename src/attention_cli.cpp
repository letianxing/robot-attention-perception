#include "robot_attention_perception/fusion.hpp"

#include <iostream>

int main(int argc, char ** argv)
{
  using namespace robot_attention_perception;
  const std::string command = argc > 1 ? argv[1] : "help";
  if (command == "algorithms") {
    for (const auto & name : AttentionFusion::availableAlgorithms()) {
      auto algorithm = AttentionFusion::createAlgorithm(name);
      std::cout << name << '\t' << algorithm->description() << '\n';
    }
    return 0;
  }
  if (command == "simulate") {
    FusionConfig config;
    if (argc > 2) {
      config.algorithm = argv[2];
    }
    AttentionFusion fusion(config);
    PerceptionFrame frame;
    frame.stamp_ms = 1000;
    frame.acoustic_tracks.push_back(
      AcousticTrack{
        "voice_1", 1000, true, 0.9F, 0.8F, 2.0F, std::nullopt, std::nullopt,
        0.2F, 0.01F, "unknown", TranscriptRecord{"voice_1", "Reachy 看一下杯子", true}});
    VisionPerson person;
    person.person_id = "owner";
    person.stamp_ms = 1000;
    person.voice_id = "voice_1";
    person.role = "owner";
    person.azimuth_deg = 3.0F;
    person.proxemic_space = "social";
    person.engagement_status = "engaged";
    person.face_visible = true;
    person.gaze_score = 0.8F;
    person.body_facing_score = 0.76F;
    person.bbox_area_ratio = 0.16F;
    person.identity_confidence = 0.9F;
    frame.vision_people.push_back(person);

    const auto state = fusion.update(frame);
    std::cout << "{"
              << "\"target_id\":\"" << state.target_id << "\","
              << "\"target_kind\":\"" << state.target_kind << "\","
              << "\"confidence\":" << state.confidence << ","
              << "\"listen\":" << (state.listen ? "true" : "false") << ","
              << "\"addressed_to_robot\":" << (state.addressed_to_robot ? "true" : "false");
    if (state.source_track_id.has_value()) {
      std::cout << ",\"source_track_id\":\"" << *state.source_track_id << "\"";
    }
    if (state.person_id.has_value()) {
      std::cout << ",\"person_id\":\"" << *state.person_id << "\"";
    }
    std::cout << "}\n";
    return 0;
  }
  std::cerr << "usage: robot_attention_cli algorithms | simulate [algorithm]\n";
  return command == "help" ? 0 : 2;
}

