#include "attention_stack/attention_core.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <perception_interfaces/msg/acoustic_tracks.hpp>
#include <perception_interfaces/msg/attention_state.hpp>
#include <perception_interfaces/msg/people.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace attention_stack
{
namespace
{
std::int64_t stampMs(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<std::int64_t>(stamp.sec) * 1000LL +
    static_cast<std::int64_t>(stamp.nanosec) / 1000000LL;
}

template<typename ArrayT>
std_msgs::msg::Float32MultiArray arrayMessage(const ArrayT & values)
{
  std_msgs::msg::Float32MultiArray message;
  message.data.assign(values.begin(), values.end());
  return message;
}
}  // namespace

class AttentionStackNode final : public rclcpp::Node
{
public:
  AttentionStackNode()
  : Node("attention_stack"), core_(loadConfig())
  {
    const auto people_topic = declare_parameter<std::string>(
      "people_topic", "/perception/vision/people");
    const auto voices_topic = declare_parameter<std::string>(
      "voices_topic", "/perception/voice/tracks");
    const auto attention_topic = declare_parameter<std::string>(
      "attention_topic", "/perception/attention/state");
    people_sub_ = create_subscription<perception_interfaces::msg::People>(
      people_topic, 10,
      [this](const perception_interfaces::msg::People::SharedPtr message) {onPeople(*message);});
    voices_sub_ = create_subscription<perception_interfaces::msg::AcousticTracks>(
      voices_topic, 10,
      [this](const perception_interfaces::msg::AcousticTracks::SharedPtr message) {onVoices(*message);});
    attention_pub_ = create_publisher<perception_interfaces::msg::AttentionState>(
      attention_topic, 10);
    attention_json_pub_ = create_publisher<std_msgs::msg::String>("/attention/state", 10);
    target_voice_pub_ = create_publisher<std_msgs::msg::String>("/attention/target_voice_id", 10);
    look_at_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/attention/look_at", 10);
    visual_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/attention/debug/s1_visual_saliency", 10);
    audio_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/attention/debug/s1_audio_saliency", 10);
    fission_pub_ = create_publisher<std_msgs::msg::Bool>("/attention/debug/s2_fission", 10);
    competition_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/attention/debug/s3_competition", 10);
    memory_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/attention/debug/s4_memory", 10);
    phase_pub_ = create_publisher<std_msgs::msg::String>("/attention/debug/s5_phase", 10);
    common_pub_ = create_publisher<std_msgs::msg::Float32>("/attention/debug/common_source_probability", 10);
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() {tick();});
  }

private:
  AttentionConfig loadConfig()
  {
    AttentionConfig config;
    config.binding_window_ms = declare_parameter<int>("binding_window_ms", 100);
    config.debounce_ms = declare_parameter<int>("debounce_ms", 300);
    config.working_memory_ms = declare_parameter<int>("working_memory_ms", 8000);
    config.habituation_tau_ms = declare_parameter<double>("habituation_tau_ms", 5000.0);
    config.common_source_prior = declare_parameter<double>("common_source_prior", 0.62);
    config.engage_threshold = declare_parameter<double>("engage_threshold", 0.52);
    config.release_threshold = declare_parameter<double>("release_threshold", 0.22);
    config.lateral_inhibition = declare_parameter<double>("lateral_inhibition", 0.18);
    config.hebbian_rate = declare_parameter<double>("hebbian_rate", 0.025);
    return config;
  }

  void onPeople(const perception_interfaces::msg::People & message)
  {
    people_.clear();
    for (const auto & item : message.people) {
      PersonCue cue;
      cue.person_id = item.person_id;
      cue.voice_id = item.voice_id;
      cue.stamp_ms = stampMs(item.observed_at);
      if (item.has_azimuth) {cue.azimuth_deg = item.azimuth_deg;}
      cue.face_confidence = item.face_confidence;
      cue.gaze_score = item.gaze_score;
      cue.facing_score = item.body_facing_score;
      cue.lip_motion = item.lip_motion;
      cue.mouth_open_ratio = item.mouth_open_ratio;
      people_.push_back(cue);
    }
  }

  void onVoices(const perception_interfaces::msg::AcousticTracks & message)
  {
    voices_.clear();
    for (const auto & item : message.tracks) {
      VoiceCue cue;
      cue.track_id = item.track_id;
      cue.stamp_ms = stampMs(item.observed_at);
      if (item.has_azimuth) {cue.azimuth_deg = item.azimuth_deg;}
      cue.active = item.voice_activity;
      cue.speech_probability = item.speech_probability;
      cue.clarity = item.clarity;
      cue.overlap_probability = item.overlap_probability;
      cue.self_echo_probability = item.self_echo_probability;
      cue.target_speaker_probability = item.target_speaker_probability;
      cue.tse_enabled = item.tse_enabled;
      cue.tse_healthy = item.tse_healthy;
      cue.tse_latency_ms = item.tse_latency_ms;
      cue.target_speech_rejected = item.target_speech_rejected;
      voices_.push_back(cue);
    }
  }

  void tick()
  {
    const auto stamp = now();
    const auto output = core_.update(stamp.nanoseconds() / 1000000LL, people_, voices_);
    perception_interfaces::msg::AttentionState typed;
    typed.header.stamp = stamp;
    typed.header.frame_id = "base_link";
    typed.algorithm = "ros4hri_native_bayesian_s1_s5";
    typed.target_id = output.target_id;
    typed.target_kind = output.target_kind;
    typed.confidence = output.confidence;
    typed.listen = output.listen;
    typed.addressed_to_robot = output.addressed_to_robot;
    typed.source_track_id = output.voice_id;
    typed.person_id = output.person_id;
    typed.has_azimuth = output.azimuth_deg.has_value();
    typed.azimuth_deg = output.azimuth_deg.value_or(0.0F);
    typed.reasons = output.reasons;
    attention_pub_->publish(typed);

    std_msgs::msg::String target_voice;
    target_voice.data = output.voice_id;
    target_voice_pub_->publish(target_voice);
    std_msgs::msg::String phase;
    phase.data = phaseName(output.phase);
    phase_pub_->publish(phase);
    std_msgs::msg::Bool fission;
    fission.data = output.fission_active;
    fission_pub_->publish(fission);
    std_msgs::msg::Float32 common;
    common.data = output.common_source_probability;
    common_pub_->publish(common);
    visual_pub_->publish(arrayMessage(output.visual_saliency));
    audio_pub_->publish(arrayMessage(output.audio_saliency));
    competition_pub_->publish(arrayMessage(output.competition));
    memory_pub_->publish(arrayMessage(output.memory_saliency));
    attention_json_pub_->publish(jsonMessage(output));
    if (output.azimuth_deg.has_value()) {look_at_pub_->publish(lookAt(*output.azimuth_deg, stamp));}
  }

  std_msgs::msg::String jsonMessage(const AttentionOutput & output) const
  {
    std::ostringstream stream;
    stream << std::setprecision(8);
    stream << "{\"stamp_ms\":" << output.stamp_ms
           << ",\"algorithm\":\"ros4hri_native_bayesian_s1_s5\""
           << ",\"target_id\":\"" << output.target_id
           << "\",\"target_kind\":\"" << output.target_kind
           << "\",\"confidence\":" << output.confidence
           << ",\"listen\":" << (output.listen ? "true" : "false")
           << ",\"addressed_to_robot\":" << (output.addressed_to_robot ? "true" : "false")
           << ",\"source_track_id\":\"" << output.voice_id
           << "\",\"person_id\":\"" << output.person_id << "\",\"azimuth_deg\":";
    if (output.azimuth_deg.has_value()) {stream << *output.azimuth_deg;} else {stream << "null";}
    const auto visual_peak = *std::max_element(output.visual_saliency.begin(), output.visual_saliency.end());
    const auto audio_peak = *std::max_element(output.audio_saliency.begin(), output.audio_saliency.end());
    stream << ",\"visual_saliency_peak\":" << visual_peak
           << ",\"audio_saliency_peak\":" << audio_peak
           << ",\"common_source_probability\":" << output.common_source_probability
           << ",\"reasons\":[";
    for (std::size_t index = 0; index < output.reasons.size(); ++index) {
      if (index > 0) {stream << ',';}
      stream << '\"' << output.reasons[index] << '\"';
    }
    stream << "],\"transcripts\":[]}";
    std_msgs::msg::String message;
    message.data = stream.str();
    return message;
  }

  geometry_msgs::msg::PointStamped lookAt(float azimuth_deg, const rclcpp::Time & stamp) const
  {
    geometry_msgs::msg::PointStamped point;
    point.header.stamp = stamp;
    point.header.frame_id = "base_link";
    const float radians = azimuth_deg * static_cast<float>(M_PI) / 180.0F;
    point.point.x = std::cos(radians);
    point.point.y = std::sin(radians);
    return point;
  }

  AttentionCore core_;
  std::vector<PersonCue> people_;
  std::vector<VoiceCue> voices_;
  rclcpp::Subscription<perception_interfaces::msg::People>::SharedPtr people_sub_;
  rclcpp::Subscription<perception_interfaces::msg::AcousticTracks>::SharedPtr voices_sub_;
  rclcpp::Publisher<perception_interfaces::msg::AttentionState>::SharedPtr attention_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr attention_json_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_voice_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr look_at_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr visual_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr audio_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fission_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr competition_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr memory_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr phase_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr common_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace attention_stack

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<attention_stack::AttentionStackNode>());
  rclcpp::shutdown();
  return 0;
}
