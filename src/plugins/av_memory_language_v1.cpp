/* av_memory_language_v1: audio-visual, working memory and the linguistic
 * context of the utterance in progress, in one attention algorithm.
 *
 * Layer 1 here is the host's original formulation: top-down terms (goal,
 * importance, motivation) are added to the bottom-up drive, and the sum is
 * squared and divisively normalised per modality. Layer 2, the per-person
 * engagement belief, lives in attention_common.hpp and is shared with every
 * other plugin, so two algorithms cannot quietly disagree about what "is this
 * person engaging me" means while claiming to differ only in their competition.
 *
 * Kept as the reference behaviour. av_memory_language_v2 replaces layer 1 with
 * the normalisation model's multiplicative attention field plus a Selective
 * Tuning inhibitory surround; compare them in the console rather than assuming
 * either is better here.
 *
 * Engineering model, not a calibrated one. No field ROC has been run.
 */
#include "attention_common.hpp"

namespace {

using namespace attention;

constexpr double kNormalisationFloor = 0.3;
constexpr double kActivationTau = 0.3;

class Engine : public EngineBase {
 protected:
  void compete(const AttentionFrameIn &frame, double dt, double gain) override;
};

void Engine::compete(const AttentionFrameIn &frame, double dt, double gain) {
  carry_forward_state(frame);

  for (const auto &entry : live_) {
    const AttentionCandidateIn &candidate = frame.candidates[entry.second];
    const CandidateState &old = previous_[entry.first];
    const double change = resolve_change(frame, entry.first, old, candidate);

    double cross = 0.0;
    const std::string link = read_id(candidate.link_id, ATTENTION_ID_MAX);
    const auto linked = live_.find(link);
    if (frame.source_audio_visual && !link.empty() && linked != live_.end()) {
      const std::string linked_modality = read_id(frame.candidates[linked->second].modality, ATTENTION_MODALITY_MAX);
      if (linked_modality != old.modality) {
        const auto neighbour = previous_.find(link);
        if (neighbour != previous_.end()) cross = unit(candidate.link_confidence) * neighbour->second.activation;
      }
    }

    /* Feedback-driven learning is not exposed through ABI v1; 0.5 is the
     * uninformed Beta(1,1) mean the host used, not a measured success rate. */
    const double learning = 0.5 * unit(candidate.observability);
    const double motivation = frame.source_internal_state ? unit(candidate.motivation) : 0.0;
    const double motive = motivation * unit(candidate.uncertainty) * learning;

    const double drive = gain * (unit(candidate.salience) + 0.6 * unit(candidate.surprise)) *
                             (1.0 - old.habituation * (1.0 - change)) * (1.0 + 0.2 * std::min(1.0, cross)) +
                         0.9 * unit(candidate.goal) + 0.6 * motive + 0.3 * unit(candidate.importance);
    const double excitation = std::max(
        0.0, drive + 0.2 * old.activation - 0.3 * old.inhibition * (1.0 - unit(candidate.goal)) * (1.0 - change));

    Drive result;
    result.excitation = excitation;
    result.change = change;
    result.input = drive;
    result.goal = unit(candidate.goal);
    result.uncertainty = unit(candidate.uncertainty);
    result.learning = learning;
    drives_[entry.first] = result;
  }

  std::map<std::string, double> denominators;
  denominators["visual"] = kNormalisationFloor * kNormalisationFloor;
  denominators["audio"] = kNormalisationFloor * kNormalisationFloor;
  for (const auto &entry : drives_) {
    denominators[previous_[entry.first].modality] += entry.second.excitation * entry.second.excitation;
  }
  for (auto &entry : drives_) {
    const double pool = denominators[previous_[entry.first].modality];
    entry.second.normalized = pool > 0.0 ? entry.second.excitation * entry.second.excitation / pool : 0.0;
  }

  decay_states(frame, dt);

  const double alpha = -std::expm1(-dt / (kActivationTau / gain));
  for (auto &entry : updated_) {
    const auto drive = drives_.find(entry.first);
    const double target = drive == drives_.end() ? 0.0 : drive->second.normalized;
    const double old = previous_[entry.first].activation;
    entry.second.activation = old + alpha * (target - old);
  }
}

const char kManifest[] =
    "{\"id\":\"av_memory_language_v1\",\"version\":\"1.1.0\",\"abi\":1,"
    "\"display_name\":\"视听 + 记忆 + 语境 v1（自上而下相加，参考实现）\","
    "\"summary\":\"自上而下项与自下而上驱动相加后平方归一化；按人累积交流意愿对数几率。作为对照的参考实现。\","
    "\"required_sources\":[\"audio_visual\"],"
    "\"optional_sources\":[\"memory_context\",\"linguistic_context\",\"cross_session_memory\",\"internal_state\"],"
    "\"outputs\":[\"attention_distribution\",\"engagement_state\",\"addressee_robot\"],"
    "\"engagement_states\":[\"IDLE\",\"OBSERVING\",\"ENGAGED\",\"INVITED\",\"EXPECTED_ANSWER\"],"
    "\"calibrated\":false,"
    "\"limits\":[\"权重为工程先验，未做现场ROC标定\",\"不授予说话或动作许可\",\"不创建未观测到的人\","
    "\"自上而下相加：刺激很弱时目标项仍可抬高响应\",\"无抑制环，方位相邻的候选会互相拉扯\","
    "\"关闭某个输入源即消融该通道证据，不替换为中性默认值\"],"
    "\"references\":["
    "\"Katzenmaier, Stiefelhagen & Schultz 2004, ICMI, doi:10.1145/1027933.1027959\","
    "\"Bohus & Horvitz 2009, SIGDIAL, aclanthology.org/W09-3933/\","
    "\"Mallidi et al. 2018, Interspeech, doi:10.21437/Interspeech.2018-1531\","
    "\"Breazeal & Scassellati 1999, context-dependent attention for a social robot\","
    "\"Eldardeer et al. 2021, Front. Neurorobot., doi:10.3389/fnbot.2021.648595\","
    "\"Itti & Koch 2001, Nat. Rev. Neurosci. 2(3):194-203\","
    "\"Desimone & Duncan 1995, Annu. Rev. Neurosci. 18:193-222\","
    "\"Ruesch et al. 2008, ICRA, multimodal saliency framework for iCub\","
    "\"Ferreira & Dias 2014, IEEE TAMD, doi:10.1109/TAMD.2014.2303072\"]}";

}  // namespace

extern "C" {

int32_t attention_plugin_abi_version(void) { return ATTENTION_PLUGIN_ABI_VERSION; }

int32_t attention_plugin_struct_size(int32_t which) {
  switch (which) {
    case ATTENTION_STRUCT_FRAME_IN: return static_cast<int32_t>(sizeof(AttentionFrameIn));
    case ATTENTION_STRUCT_CANDIDATE_IN: return static_cast<int32_t>(sizeof(AttentionCandidateIn));
    case ATTENTION_STRUCT_FRAME_OUT: return static_cast<int32_t>(sizeof(AttentionFrameOut));
    case ATTENTION_STRUCT_CANDIDATE_OUT: return static_cast<int32_t>(sizeof(AttentionCandidateOut));
    case ATTENTION_STRUCT_KEY_VALUE: return static_cast<int32_t>(sizeof(AttentionKeyValue));
    default: return -1;
  }
}

const char *attention_plugin_manifest(void) { return kManifest; }

void *attention_plugin_create(void) { return new Engine(); }

void attention_plugin_destroy(void *handle) { delete static_cast<Engine *>(handle); }

const AttentionFrameOut *attention_plugin_update(void *handle, const AttentionFrameIn *frame) {
  if (handle == nullptr || frame == nullptr) return nullptr;
  return static_cast<Engine *>(handle)->update(*frame);
}

}  // extern "C"
