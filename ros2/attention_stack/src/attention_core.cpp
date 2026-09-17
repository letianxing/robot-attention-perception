#include "attention_stack/attention_core.hpp"

#include <algorithm>
#include <cmath>

namespace attention_stack
{
namespace
{
float clamp(float value, float low = 0.0F, float high = 1.0F)
{
  return std::max(low, std::min(value, high));
}

float angularDistance(float left, float right)
{
  float delta = std::fmod(std::abs(left - right), 360.0F);
  return std::min(delta, 360.0F - delta);
}
}  // namespace

AttentionCore::AttentionCore(AttentionConfig config)
: config_(config)
{
}

AttentionOutput AttentionCore::update(
  std::int64_t stamp_ms,
  const std::vector<PersonCue> & people,
  const std::vector<VoiceCue> & voices)
{
  AttentionOutput output;
  output.stamp_ms = stamp_ms;
  bool directional_evidence = false;
  int audio_onsets = 0;
  for (const auto & person : people) {
    if (!person.azimuth_deg.has_value()) {continue;}
    directional_evidence = true;
    const int bin = azimuthBin(*person.azimuth_deg);
    const float saliency = clamp(
      0.35F * person.face_confidence + 0.35F * person.gaze_score +
      0.20F * person.facing_score + 0.10F * (person.lip_motion ? 1.0F : 0.0F));
    output.visual_saliency[bin] = std::max(output.visual_saliency[bin], saliency);
  }
  for (const auto & voice : voices) {
    const bool previous = previous_voice_activity_[voice.track_id];
    if (voice.active && !previous) {++audio_onsets;}
    previous_voice_activity_[voice.track_id] = voice.active;
    if (!voice.active) {continue;}
      const float saliency = clamp(
      (0.55F * voice.speech_probability + 0.45F * voice.clarity) *
      (voice.tse_enabled ? voice.target_speaker_probability : 1.0F) -
      0.25F * voice.overlap_probability - 0.70F * voice.self_echo_probability);
    if (voice.azimuth_deg.has_value()) {
      directional_evidence = true;
      const int bin = azimuthBin(*voice.azimuth_deg);
      output.audio_saliency[bin] = std::max(output.audio_saliency[bin], saliency);
    } else {
      // A single microphone has no bearing, but it must still be able to
      // open the listening gate.  The previous 0.32 scale made ordinary
      // far-field speech fall below engage_threshold even when VAD was true.
      // Keep it broad (no fabricated direction), but retain most of the
      // measured speech saliency; spatial arrays still use the directional
      // branch above.
      for (float & value : output.audio_saliency) {value = std::max(value, saliency * 0.80F);}
    }
  }

  float best_common = 0.0F;
  for (const auto & person : people) {
    for (const auto & voice : voices) {
      if (!voice.active || voice.target_speech_rejected) {continue;}
      best_common = std::max(best_common, commonSourceProbability(person, voice));
    }
  }
  output.common_source_probability = best_common;
  const int visual_events = static_cast<int>(std::count_if(
    people.begin(), people.end(), [](const PersonCue & cue) {return cue.lip_motion;}));
  output.fission_active = audio_onsets >= 2 && visual_events == 1;

  std::array<float, kBins> raw{};
  for (std::size_t index = 0; index < kBins; ++index) {
    const float integrated = output.visual_saliency[index] + output.audio_saliency[index];
    const float segregated = std::max(output.visual_saliency[index], output.audio_saliency[index]);
    raw[index] = best_common * integrated + (1.0F - best_common) * segregated;
    if (output.fission_active) {raw[index] *= 0.82F;}
  }
  for (std::size_t index = 0; index < kBins; ++index) {
    const std::size_t left = (index + kBins - 1) % kBins;
    const std::size_t right = (index + 1) % kBins;
    const float inhibited = raw[index] - config_.lateral_inhibition *
      0.5F * (raw[left] + raw[right]);
    output.competition[index] = std::max(0.0F, inhibited);
  }

  const float dt_ms = last_update_ms_ == 0 ? 20.0F :
    static_cast<float>(std::max<std::int64_t>(1, stamp_ms - last_update_ms_));
  last_update_ms_ = stamp_ms;
  const float decay = std::exp(-dt_ms / std::max(1.0F, config_.habituation_tau_ms));
  for (std::size_t index = 0; index < kBins; ++index) {
    habituation_[index] *= decay;
    habituation_[index] = clamp(habituation_[index] + 0.015F * output.competition[index]);
    output.memory_saliency[index] = output.competition[index] * (1.0F - 0.22F * habituation_[index]);
  }
  if (last_bin_ >= 0 && stamp_ms - last_target_ms_ <= config_.working_memory_ms) {
    const float memory = 0.16F * (1.0F - static_cast<float>(stamp_ms - last_target_ms_) /
      static_cast<float>(config_.working_memory_ms));
    output.memory_saliency[static_cast<std::size_t>(last_bin_)] += std::max(0.0F, memory);
  }
  for (std::size_t visual_bin = 0; visual_bin < kBins; ++visual_bin) {
    for (std::size_t audio_bin = 0; audio_bin < kBins; ++audio_bin) {
      hebbian_[visual_bin][audio_bin] *= 0.999F;
      hebbian_[visual_bin][audio_bin] += config_.hebbian_rate *
        output.visual_saliency[visual_bin] * output.audio_saliency[audio_bin];
    }
  }

  const auto winner = std::max_element(output.memory_saliency.begin(), output.memory_saliency.end());
  const int winner_bin = static_cast<int>(std::distance(output.memory_saliency.begin(), winner));
  output.confidence = clamp(*winner);
  updatePhase(output.confidence, winner_bin, stamp_ms);
  output.phase = phase_;
  output.listen = phase_ == AttentionPhase::ENGAGED;
    output.azimuth_deg = output.confidence > 0.0F && directional_evidence ?
    std::optional<float>(binAzimuth(winner_bin)) : std::nullopt;
  if (output.confidence > 0.0F) {
    const PersonCue * best_person = nullptr;
    float person_delta = 999.0F;
    for (const auto & person : people) {
      if (!person.azimuth_deg.has_value()) {continue;}
      const float delta = angularDistance(*person.azimuth_deg, binAzimuth(winner_bin));
      if (delta < person_delta) {person_delta = delta; best_person = &person;}
    }
    const VoiceCue * best_voice = nullptr;
    float voice_delta = 999.0F;
    for (const auto & voice : voices) {
      if (!voice.active) {continue;}
      const float delta = voice.azimuth_deg.has_value() ?
        angularDistance(*voice.azimuth_deg, binAzimuth(winner_bin)) : 0.0F;
      if (delta < voice_delta) {voice_delta = delta; best_voice = &voice;}
    }
    if (best_person != nullptr && (best_common >= 0.48F || voices.size() == 1)) {
      output.person_id = best_person->person_id;
      output.target_id = "person:" + best_person->person_id;
      output.target_kind = "person";
      output.addressed_to_robot = output.listen &&
        (best_person->gaze_score >= 0.58F || best_person->lip_motion);
    }
    if (best_voice != nullptr) {
      output.voice_id = best_voice->track_id;
      if (output.target_kind == "none") {
        output.target_id = "sound:" + best_voice->track_id;
        output.target_kind = "sound_source";
      }
    }
    if (best_person != nullptr && best_voice != nullptr &&
      best_person->azimuth_deg.has_value() && best_voice->azimuth_deg.has_value())
    {
      const float visual_weight = std::max(0.01F, best_person->face_confidence);
      const float audio_weight = std::max(
        0.01F, 0.5F * best_voice->speech_probability + 0.5F * best_voice->clarity);
      output.azimuth_deg = ((*best_person->azimuth_deg * visual_weight) +
        (*best_voice->azimuth_deg * audio_weight)) / (visual_weight + audio_weight);
    } else if (best_person != nullptr && best_person->azimuth_deg.has_value()) {
      output.azimuth_deg = best_person->azimuth_deg;
    } else if (best_voice != nullptr && best_voice->azimuth_deg.has_value()) {
      output.azimuth_deg = best_voice->azimuth_deg;
    }
  }
  output.reasons = {
    "s1_egocentric_saliency", "s2_temporal_binding", "s3_wta_lateral_inhibition",
    "s4_habituation_working_memory_hebbian", "s5_" + phaseName(phase_),
    "bayesian_common:" + std::to_string(best_common)};
  if (output.fission_active) {output.reasons.push_back("fission_active");}
  return output;
}

int AttentionCore::azimuthBin(float azimuth_deg)
{
  float wrapped = std::fmod(azimuth_deg + 180.0F, 360.0F);
  if (wrapped < 0.0F) {wrapped += 360.0F;}
  return std::min(35, static_cast<int>(wrapped / 10.0F));
}

float AttentionCore::binAzimuth(int bin)
{
  return -175.0F + 10.0F * static_cast<float>(bin);
}

float AttentionCore::commonSourceProbability(const PersonCue & person, const VoiceCue & voice) const
{
  const float temporal_delta = static_cast<float>(std::abs(person.stamp_ms - voice.stamp_ms));
  const float temporal = std::exp(-0.5F * std::pow(
    temporal_delta / std::max(1, config_.binding_window_ms), 2.0F));
  float spatial = 0.72F;
  if (person.azimuth_deg.has_value() && voice.azimuth_deg.has_value()) {
    spatial = std::exp(-0.5F * std::pow(
      angularDistance(*person.azimuth_deg, *voice.azimuth_deg) / 18.0F, 2.0F));
  }
  const float reliability = clamp(person.face_confidence) *
    clamp(0.5F * voice.speech_probability + 0.5F * voice.clarity) *
    (voice.tse_enabled ? clamp(voice.target_speaker_probability) : 1.0F);
  const float likelihood = spatial * temporal * (0.35F + 0.65F * reliability);
  const float numerator = config_.common_source_prior * likelihood;
  return numerator / std::max(
    1e-6F, numerator + (1.0F - config_.common_source_prior) * 0.22F);
}

void AttentionCore::updatePhase(float confidence, int winner_bin, std::int64_t stamp_ms)
{
  if (phase_since_ms_ == 0) {phase_since_ms_ = stamp_ms;}
  switch (phase_) {
    case AttentionPhase::IDLE:
      if (confidence >= config_.engage_threshold) {
        phase_ = AttentionPhase::ORIENTING; phase_since_ms_ = stamp_ms; last_bin_ = winner_bin;
      }
      break;
    case AttentionPhase::ORIENTING:
      if (confidence < config_.release_threshold) {phase_ = AttentionPhase::IDLE; phase_since_ms_ = stamp_ms;}
      else if (winner_bin != last_bin_) {last_bin_ = winner_bin; phase_since_ms_ = stamp_ms;}
      else if (stamp_ms - phase_since_ms_ >= config_.debounce_ms) {
        phase_ = AttentionPhase::ENGAGED; phase_since_ms_ = stamp_ms; last_target_ms_ = stamp_ms;
      }
      break;
    case AttentionPhase::ENGAGED:
      if (confidence < config_.release_threshold) {phase_ = AttentionPhase::RELEASE; phase_since_ms_ = stamp_ms;}
      else {last_bin_ = winner_bin; last_target_ms_ = stamp_ms;}
      break;
    case AttentionPhase::RELEASE:
      if (confidence >= config_.engage_threshold) {phase_ = AttentionPhase::ORIENTING; phase_since_ms_ = stamp_ms;}
      else if (stamp_ms - phase_since_ms_ >= config_.debounce_ms) {phase_ = AttentionPhase::IDLE; phase_since_ms_ = stamp_ms;}
      break;
  }
}

std::string phaseName(AttentionPhase phase)
{
  switch (phase) {
    case AttentionPhase::IDLE: return "IDLE";
    case AttentionPhase::ORIENTING: return "ORIENTING";
    case AttentionPhase::ENGAGED: return "ENGAGED";
    case AttentionPhase::RELEASE: return "RELEASE";
  }
  return "IDLE";
}

}  // namespace attention_stack
