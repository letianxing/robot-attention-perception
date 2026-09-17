/* av_memory_language_v2: same evidence and same engagement layer as v1, with
 * layer 1 rebuilt on two models v1 only approximated.
 *
 *   Normalisation model of attention (Reynolds & Heeger 2009).
 *     R = A * E / (S + sigma), where E is the bottom-up stimulus drive, A is a
 *     multiplicative attention field built from the top-down terms, and S is the
 *     suppressive drive. The pool is the same modality plus a fraction of the
 *     other one, so attention really is a limited resource across the senses.
 *     v1 added the top-down
 *     terms to E instead, which lets a goal raise the response of a candidate
 *     that is barely being sensed. Here attention modulates what is measured and
 *     cannot manufacture a response out of nothing.
 *
 *   Selective Tuning (Tsotsos et al.).
 *     After a winner is selected, its surround is inhibited so the selected
 *     object is isolated from nearby distractors. Implemented as an angular
 *     surround around the current focus, which is what stops two people standing
 *     close together from pulling the focus back and forth. Candidates without a
 *     known azimuth are left alone rather than guessed at.
 *
 * Both are engineering adaptations of those models, not reimplementations of
 * their published architectures, and neither has been calibrated here. This is
 * offered as the comparison against v1, not as a measured improvement: switch
 * between them in the console and look at target oscillation, false barge-in and
 * identity binding on the same recording before preferring either.
 */
#include "attention_common.hpp"

namespace {

using namespace attention;

/* Reynolds-Heeger */
constexpr double kSemiSaturation = 0.3;  /* sigma: keeps a lone weak candidate from saturating */
constexpr double kFieldGoal = 1.6;
constexpr double kFieldImportance = 0.6;
constexpr double kFieldMotive = 1.0;
constexpr double kActivationTau = 0.3;

/* Selective Tuning surround */
constexpr double kSurroundDegrees = 25.0;
constexpr double kSurroundStrength = 0.6;

/* How much the other modality counts in the normalisation pool. Attention is a
 * limited resource across senses, so a loud voice does reduce how much of it a
 * face can hold; but eyes and ears are separate effectors and each keeps its own
 * focus, so the pools are coupled rather than merged. */
constexpr double kCrossModalPool = 0.35;

double angular_distance(double left, double right) {
  double delta = std::fmod(left - right + 180.0, 360.0);
  if (delta < 0.0) delta += 360.0;
  return std::fabs(delta - 180.0);
}

class Engine : public EngineBase {
 protected:
  void compete(const AttentionFrameIn &frame, double dt, double gain) override;

 private:
  void suppress_surround(const AttentionFrameIn &frame);
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

    const double learning = 0.5 * unit(candidate.observability);
    const double motivation = frame.source_internal_state ? unit(candidate.motivation) : 0.0;
    const double motive = motivation * unit(candidate.uncertainty) * learning;

    /* Stimulus drive: only what is actually being sensed right now. */
    const double stimulus = gain * (unit(candidate.salience) + 0.6 * unit(candidate.surprise)) *
                            (1.0 - old.habituation * (1.0 - change)) * (1.0 + 0.2 * std::min(1.0, cross));
    /* Attention field: a gain of at least 1, never a source of response. */
    const double field = 1.0 + kFieldGoal * unit(candidate.goal) + kFieldImportance * unit(candidate.importance) +
                         kFieldMotive * motive;
    const double attended = stimulus * field;
    const double excitation = std::max(
        0.0, attended - 0.3 * old.inhibition * (1.0 - unit(candidate.goal)) * (1.0 - change));

    Drive result;
    result.excitation = excitation;
    result.change = change;
    result.input = attended;
    result.goal = unit(candidate.goal);
    result.uncertainty = unit(candidate.uncertainty);
    result.learning = learning;
    drives_[entry.first] = result;
  }

  /* Suppressive drive: a crowded modality divides its response between its
   * members, and the other modality contributes a fraction of its own drive, so
   * a busy soundscape really does cost visual candidates some response instead
   * of the two senses being normalised in isolation. */
  std::map<std::string, double> pools;
  for (const auto &entry : drives_) pools[previous_[entry.first].modality] += entry.second.excitation;
  for (auto &entry : drives_) {
    const std::string &modality = previous_[entry.first].modality;
    const std::string other = modality == "visual" ? "audio" : "visual";
    const double pool = pools[modality] + kCrossModalPool * pools[other];
    entry.second.normalized = entry.second.excitation / (pool + kSemiSaturation);
  }

  suppress_surround(frame);
  decay_states(frame, dt);

  const double alpha = -std::expm1(-dt / (kActivationTau / gain));
  for (auto &entry : updated_) {
    const auto drive = drives_.find(entry.first);
    const double target = drive == drives_.end() ? 0.0 : drive->second.normalized;
    const double old = previous_[entry.first].activation;
    entry.second.activation = old + alpha * (target - old);
  }
}

void Engine::suppress_surround(const AttentionFrameIn &frame) {
  /* The beam centre is the focus selected on the previous cycle: using this
   * cycle's winner would make selection depend on its own outcome. */
  for (const char *modality : {"visual", "audio"}) {
    const std::string &focus = std::strcmp(modality, "visual") == 0 ? focus_visual_ : focus_audio_;
    const auto centre = live_.find(focus);
    if (focus.empty() || centre == live_.end()) continue;
    double centre_azimuth = 0.0;
    if (!extra_raw(frame.candidates[centre->second], ATTENTION_EXTRA_AZIMUTH_DEG, &centre_azimuth)) continue;

    for (auto &entry : drives_) {
      if (entry.first == focus) continue;
      if (previous_[entry.first].modality != modality) continue;
      const auto other = live_.find(entry.first);
      if (other == live_.end()) continue;
      double azimuth = 0.0;
      if (!extra_raw(frame.candidates[other->second], ATTENTION_EXTRA_AZIMUTH_DEG, &azimuth)) continue;
      const double delta = angular_distance(azimuth, centre_azimuth);
      if (delta >= kSurroundDegrees) continue;
      entry.second.normalized *= 1.0 - kSurroundStrength * (1.0 - delta / kSurroundDegrees);
    }
  }
}

const char kManifest[] =
    "{\"id\":\"av_memory_language_v2\",\"version\":\"1.0.0\",\"abi\":1,"
    "\"display_name\":\"视听 + 记忆 + 语境 v2（乘性注意场 + 跨模态池 + 抑制环）\","
    "\"summary\":\"自上而下作为乘性注意场调制自下而上刺激后再归一化（Reynolds-Heeger），归一化池跨模态耦合；"
    "选中目标的方位邻域被抑制（Selective Tuning），减少相邻候选来回拉扯。交流意愿层与 v1 相同。\","
    "\"required_sources\":[\"audio_visual\"],"
    "\"optional_sources\":[\"memory_context\",\"linguistic_context\",\"cross_session_memory\",\"internal_state\"],"
    "\"outputs\":[\"attention_distribution\",\"engagement_state\",\"addressee_robot\"],"
    "\"engagement_states\":[\"IDLE\",\"OBSERVING\",\"ENGAGED\",\"INVITED\",\"EXPECTED_ANSWER\"],"
    "\"calibrated\":false,"
    "\"limits\":[\"权重为工程先验，未做现场ROC标定；与 v1 的优劣未经现场对比\","
    "\"是对两篇模型的工程改写，不是其发表架构的复现\","
    "\"抑制环需要候选带方位；无方位时不抑制，而不是猜一个\","
    "\"跨模态耦合系数 0.35 是工程先验；视听仍各自保留独立焦点，因为眼睛和耳朵是两个执行器\","
    "\"注意场只调制已测到的刺激，刺激为零时目标项不再产生响应（与 v1 的相加式不同）\","
    "\"不授予说话或动作许可\",\"不创建未观测到的人\","
    "\"关闭某个输入源即消融该通道证据，不替换为中性默认值\"],"
    "\"references\":["
    "\"Reynolds & Heeger 2009, The normalization model of attention, Neuron 61(2):168-185\","
    "\"Tsotsos et al. 1995, Modeling visual attention via selective tuning, Artif. Intell. 78:507-545\","
    "\"Itti & Koch 2001, Nat. Rev. Neurosci. 2(3):194-203\","
    "\"Desimone & Duncan 1995, Annu. Rev. Neurosci. 18:193-222\","
    "\"Katzenmaier, Stiefelhagen & Schultz 2004, ICMI, doi:10.1145/1027933.1027959\","
    "\"Bohus & Horvitz 2009, SIGDIAL, aclanthology.org/W09-3933/\","
    "\"Mallidi et al. 2018, Interspeech, doi:10.21437/Interspeech.2018-1531\","
    "\"Eldardeer et al. 2021, Front. Neurorobot., doi:10.3389/fnbot.2021.648595\"]}";

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
