/* Shared machinery for attention algorithm plugins.
 *
 * A plugin supplies layer 1 — how candidates compete for the focus — and
 * inherits layer 2, the per-person engagement belief, plus the reporting that
 * turns both into an AttentionFrameOut. Keeping layer 2 in one place means two
 * algorithms cannot quietly disagree about what "is this person engaging me"
 * means while claiming to differ only in their competition.
 *
 * Engineering model, not a calibrated one. Nothing here grants speech or motion
 * permission, creates a person that was not observed, or rewrites an identity.
 */
#ifndef ROBOT_ATTENTION_PERCEPTION_ATTENTION_COMMON_HPP
#define ROBOT_ATTENTION_PERCEPTION_ATTENTION_COMMON_HPP

#include "robot_attention_perception/attention_plugin_abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace attention {

/* Engagement layer, shared by every algorithm. */
constexpr double kEngageTau = 1.2;      /* seconds of evidence integration */
/* Belief may decay at the true rate, but it may not accumulate for time nobody
 * observed. A stalled loop or a long gap between frames must not let the
 * integrator jump straight to steady state and skip the sustained evidence the
 * thresholds are built on. Forgetting uses the real dt; rising does not. */
constexpr double kMaxRisingDt = 1.0;
constexpr double kWeightAudioVisual = 0.76;
constexpr double kWeightMemory = 2.76;
constexpr double kWeightAgainst = 1.6;
constexpr double kWeightFamiliarity = 0.15;
/* An exchange that is already running does not end because the camera lost
 * them. Somebody who was addressing us moments ago and speaks again is still
 * addressing us, on or off camera. Decayed by the host, so this fades over
 * roughly fifteen seconds rather than latching. */
constexpr double kWeightOpenExchange = 0.40;
constexpr double kCouplingFloor = 0.4;  /* unattended evidence still counts a little */
constexpr double kPhasicCall = 1.5;
constexpr double kPhasicAnswer = 1.4;   /* scaled by identity, so a confident match still clears the bar */
constexpr double kPhasicTopic = 0.6;
constexpr double kPhasicQuestion = 0.35;
constexpr double kPhasicOther = 1.4;
constexpr double kPhasicSyncOnset = 1.3;  /* looking at us and starting to speak */
constexpr double kSpeakingToUs = 0.95;    /* speech counts in proportion to facing us */
constexpr double kPhasicInvite = 2.2;     /* waving or beckoning at the robot */
constexpr double kPhasicReject = 2.4;     /* being waved off outweighs being waved at */
constexpr int64_t kGestureReasonMs = 3000;  /* how long a gesture stays in the log */
constexpr double kLogitLimit = 4.0;
/* Unbroken mutual gaze from somebody who is not speaking. Dyadic mutual gaze is
 * comfortable for about 3.3 s on average (Binetti et al. 2016); past that it
 * stops being ordinary looking and becomes a bid for attention, which is why
 * the term only starts there and needs several more seconds to saturate. It is
 * scaled by recognition confidence: a flickering detection that is not reliably
 * the same person cannot stare at anybody. */
constexpr double kWeightSustainedGaze = 1.2;
constexpr double kGazeInviteMs = 3300.0;
constexpr double kGazeSaturateMs = 8000.0;
/* Once the look has been answered the evidence is held briefly so downstream
 * sees a state rather than a single frame, then suppressed until they look
 * away and back. Somebody working in front of the robot is not asked twice. */
constexpr int64_t kGazeInviteHoldMs = 2000;
constexpr double kEngageThreshold = 0.72;
constexpr double kObserveThreshold = 0.55;
constexpr double kEvidenceBit = 0.15;

/* Focus selection, shared so switching algorithms does not change hysteresis. */
constexpr double kFocusEnter = 0.25;
constexpr double kFocusMargin = 0.08;
constexpr double kFocusHold = 0.15;

inline double unit(double value) {
  if (!std::isfinite(value)) return 0.0;
  return std::min(1.0, std::max(0.0, value));
}

inline double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

inline void copy_id(char *destination, std::size_t size, const std::string &source) {
  std::memset(destination, 0, size);
  std::strncpy(destination, source.c_str(), size - 1);
}

inline std::string read_id(const char *source, std::size_t size) {
  return std::string(source, ::strnlen(source, size));
}

/* Inputs added after ABI v1 froze travel in the generic key/value array, so a
 * new evidence source does not force every plugin to be recompiled. An absent
 * key means the host did not supply it, which is also how a disabled source
 * looks — the frame's source flags and the console report which it was. */
inline bool extra_raw(const AttentionCandidateIn &candidate, const char *key, double *out) {
  if (candidate.extra == nullptr) return false;
  for (int32_t index = 0; index < candidate.extra_count; ++index) {
    if (std::strncmp(candidate.extra[index].key, key, ATTENTION_ID_MAX) == 0) {
      if (!std::isfinite(candidate.extra[index].value)) return false;
      *out = candidate.extra[index].value;
      return true;
    }
  }
  return false;
}

inline double extra_of(const AttentionCandidateIn &candidate, const char *key) {
  double value = 0.0;
  return extra_raw(candidate, key, &value) ? unit(value) : 0.0;
}

/* Frame-level tunables, read from the generic key/value array the ABI reserves
 * for exactly this. Nothing here changes what the algorithm is; it changes the
 * numbers the field keeps disagreeing with, without a rebuild and without
 * reordering a struct. Absent key means "use the compiled default", so an older
 * host and a newer plugin still work. */
struct Tunables {
  double engage_threshold = kEngageThreshold;
  double observe_threshold = kObserveThreshold;
  double engage_tau = kEngageTau;
  double weight_audio_visual = kWeightAudioVisual;
  double weight_memory = kWeightMemory;
  double weight_against = kWeightAgainst;
  double sustained_gaze = kWeightSustainedGaze;
  double gaze_invite_ms = kGazeInviteMs;
  double gaze_saturate_ms = kGazeSaturateMs;
};

inline bool frame_extra(const AttentionFrameIn &frame, const char *key, double *out) {
  if (frame.extra == nullptr) return false;
  for (int32_t index = 0; index < frame.extra_count; ++index) {
    if (std::strncmp(frame.extra[index].key, key, ATTENTION_ID_MAX) == 0) {
      if (!std::isfinite(frame.extra[index].value)) return false;
      *out = frame.extra[index].value;
      return true;
    }
  }
  return false;
}

inline Tunables tunables_of(const AttentionFrameIn &frame) {
  Tunables tuned;
  frame_extra(frame, "tune_engage_threshold", &tuned.engage_threshold);
  frame_extra(frame, "tune_observe_threshold", &tuned.observe_threshold);
  frame_extra(frame, "tune_engage_tau", &tuned.engage_tau);
  frame_extra(frame, "tune_weight_audio_visual", &tuned.weight_audio_visual);
  frame_extra(frame, "tune_weight_memory", &tuned.weight_memory);
  frame_extra(frame, "tune_weight_against", &tuned.weight_against);
  frame_extra(frame, "tune_sustained_gaze", &tuned.sustained_gaze);
  frame_extra(frame, "tune_gaze_invite_ms", &tuned.gaze_invite_ms);
  frame_extra(frame, "tune_gaze_saturate_ms", &tuned.gaze_saturate_ms);
  if (tuned.engage_tau < .05) tuned.engage_tau = .05;
  if (tuned.gaze_saturate_ms <= tuned.gaze_invite_ms) tuned.gaze_saturate_ms = tuned.gaze_invite_ms + 100.;
  return tuned;
}

struct CandidateState {
  double activation = 0.0;
  double habituation = 0.0;
  double inhibition = 0.0;
  double change = 0.0;
  std::string modality;
  int64_t last_seen = 0;
};

struct PersonState {
  double logit = 0.0;
  std::string last_event;
  int64_t last_seen = 0;
  /* A gesture is applied once but has to stay explainable afterwards, or the
   * log cannot say why the belief jumped. */
  int64_t invited_ms = 0;
  int64_t rejected_ms = 0;
  /* How long this person has been looking at us without speaking, and whether
   * that look has already been answered. One invitation per episode of gaze:
   * somebody working in front of the robot must not be asked every few seconds. */
  double gaze_hold_ms = 0.0;
  int64_t gaze_invited_ms = 0;
};

/* What layer 1 computed for one live candidate this cycle. */
struct Drive {
  double excitation = 0.0;  /* what enters the normalisation pool */
  double change = 0.0;
  double input = 0.0;       /* pre-normalisation drive, reported for diagnosis */
  double goal = 0.0;
  double uncertainty = 0.0;
  double learning = 0.0;
  double normalized = 0.0;  /* layer 1 writes the final normalised response here */
};

/* Evidence for one person gathered across every candidate that resolves to
 * them, so a face and a voice of the same person reinforce one belief. */
struct PersonEvidence {
  double audio_visual = 0.0;
  double memory = 0.0;
  double against = 0.0;
  double activation = 0.0;
  double voice_activity = 0.0;
  double identity = 0.0;
  double gaze = 0.0;
  double body_facing = 0.0;
  double lip_sync = 0.0;
  double sync_onset = 0.0;
  double gesture_invite = 0.0;
  double gesture_reject = 0.0;
  double sustained_gaze = 0.0;
  double participant = 0.0;
  double familiarity = 0.0;
  double open_exchange = 0.0;
  double expected_answer = 0.0;
  double dialogue_target = 0.0;
  double directed_call = 0.0;
  double question = 0.0;
  double answer_continuation = 0.0;
  double topic_continuation = 0.0;
  double addresses_other = 0.0;
  double backchannel = 0.0;
  double motivation = 0.0;
  std::string event_id;
  std::vector<std::size_t> candidates;
};

class EngineBase {
 public:
  virtual ~EngineBase() = default;

  const AttentionFrameOut *update(const AttentionFrameIn &frame);

 protected:
  /* Layer 1: fill drives_ (including `normalized`) and updated_ for this cycle.
   * dt is seconds since the previous accepted frame, gain the arousal factor. */
  virtual void compete(const AttentionFrameIn &frame, double dt, double gain) = 0;

  /* Shared helpers layer 1 implementations use. */
  void carry_forward_state(const AttentionFrameIn &frame);
  double resolve_change(const AttentionFrameIn &frame, const std::string &key, const CandidateState &old,
                        const AttentionCandidateIn &candidate);
  void decay_states(const AttentionFrameIn &frame, double dt);
  void select_focus();

  std::map<std::string, CandidateState> state_;
  std::map<std::string, CandidateState> previous_;
  std::map<std::string, CandidateState> updated_;
  std::map<std::string, std::size_t> live_;
  std::map<std::string, Drive> drives_;
  std::set<std::string> seen_changes_;
  std::set<std::string> new_changes_;
  std::string focus_visual_;
  std::string focus_audio_;
  int64_t stamp_ms_ = -1;
  int32_t transitions_ = 0;

 private:
  void accumulate_engagement(const AttentionFrameIn &frame, double dt);
  void report(const AttentionFrameIn &frame);

  std::map<std::string, PersonState> people_;
  Tunables tuned_;
  std::map<std::string, PersonEvidence> evidence_;
  std::vector<AttentionCandidateOut> output_;
  AttentionFrameOut frame_out_{};
};

inline void EngineBase::carry_forward_state(const AttentionFrameIn &frame) {
  previous_ = state_;
  for (const auto &entry : live_) {
    if (previous_.find(entry.first) != previous_.end()) continue;
    CandidateState fresh;
    fresh.modality = read_id(frame.candidates[entry.second].modality, ATTENTION_MODALITY_MAX);
    fresh.last_seen = frame.stamp_ms;
    previous_[entry.first] = fresh;
  }
}

inline double EngineBase::resolve_change(const AttentionFrameIn &, const std::string &, const CandidateState &old,
                                         const AttentionCandidateIn &candidate) {
  const std::string change_id = read_id(candidate.change_id, ATTENTION_ID_MAX);
  if (change_id.empty() || seen_changes_.find(change_id) != seen_changes_.end()) return old.change;
  new_changes_.insert(change_id);
  return 1.0;
}

/* Habituation, recovery and return inhibition are properties of a tracked
 * candidate rather than of any one competition rule, so every algorithm shares
 * the same exact-solution decay. */
inline void EngineBase::decay_states(const AttentionFrameIn &frame, double dt) {
  seen_changes_.insert(new_changes_.begin(), new_changes_.end());
  new_changes_.clear();
  updated_.clear();
  for (const auto &entry : previous_) {
    const std::string &key = entry.first;
    const CandidateState &old = entry.second;
    const bool present = live_.find(key) != live_.end();
    const Drive drive = present ? drives_[key] : Drive{0.0, old.change, 0.0, 0.0, 0.0, 0.0, 0.0};

    CandidateState next;
    next.modality = old.modality;
    next.activation = old.activation;  /* layer 1 already advanced this */
    const double exposure = present ? 1.0 : 0.0;
    const double up = exposure * (1.0 - drive.change) / 10.0;
    const double down = (1.0 - exposure) / 30.0 + drive.change / 0.5;
    const double rate = up + down;
    const double equilibrium = rate > 0.0 ? up / rate : 0.0;
    next.habituation =
        rate > 0.0 ? unit(equilibrium + (old.habituation - equilibrium) * std::exp(-rate * dt)) : old.habituation;
    next.inhibition = old.inhibition * std::exp(-dt / 1.5);
    next.change = drive.change * std::exp(-dt);
    next.last_seen = present ? frame.stamp_ms : old.last_seen;
    updated_[key] = next;
  }
}

inline void EngineBase::select_focus() {
  for (const char *modality : {"visual", "audio"}) {
    std::string &focus = std::strcmp(modality, "visual") == 0 ? focus_visual_ : focus_audio_;
    std::vector<std::pair<double, std::string>> ranked;
    for (const auto &entry : updated_) {
      if (entry.second.modality != modality) continue;
      if (live_.find(entry.first) == live_.end()) continue;
      ranked.emplace_back(entry.second.activation, entry.first);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto &left, const auto &right) { return right < left; });
    const std::string old_focus = focus;
    const bool held = live_.count(old_focus) > 0 && updated_[old_focus].activation >= kFocusHold;
    std::string next = held ? old_focus : std::string();
    if (!ranked.empty()) {
      const double winner = ranked[0].first;
      const double runner = ranked.size() > 1 ? ranked[1].first : 0.0;
      const bool strong = winner >= kFocusEnter && winner - runner >= kFocusMargin;
      const bool allowed = !held || ranked[0].second == old_focus || winner - updated_[old_focus].activation >= kFocusMargin;
      if (strong && allowed) next = ranked[0].second;
    }
    if (next != old_focus) {
      ++transitions_;
      if (!old_focus.empty() && updated_.count(old_focus)) updated_[old_focus].inhibition = 1.0;
    }
    focus = next;
  }
}

inline void EngineBase::accumulate_engagement(const AttentionFrameIn &frame, double dt) {
  tuned_ = tunables_of(frame);
  evidence_.clear();
  for (const auto &entry : live_) {
    const AttentionCandidateIn &candidate = frame.candidates[entry.second];
    const std::string person = read_id(candidate.person_id, ATTENTION_ID_MAX);
    if (person.empty() || person == "unknown" || person == "robot") continue;
    PersonEvidence &item = evidence_[person];
    item.candidates.push_back(entry.second);
    item.activation = std::max(item.activation, updated_[entry.first].activation);
    item.identity = std::max(item.identity, unit(candidate.identity_confidence));
    item.gaze = std::max(item.gaze, unit(candidate.gaze));
    item.body_facing = std::max(item.body_facing, unit(candidate.body_facing));
    item.lip_sync = std::max(item.lip_sync, unit(candidate.lip_sync));
    item.voice_activity = std::max(item.voice_activity, unit(candidate.voice_activity));
    item.participant = std::max(item.participant, unit(candidate.memory_participant) * unit(candidate.memory_recency));
    item.familiarity = std::max(item.familiarity, extra_of(candidate, ATTENTION_EXTRA_MEMORY_FAMILIARITY));
    item.open_exchange = std::max(item.open_exchange, extra_of(candidate, ATTENTION_EXTRA_OPEN_EXCHANGE));
    item.sync_onset = std::max(item.sync_onset, extra_of(candidate, ATTENTION_EXTRA_AV_SYNC_ONSET));
    item.gesture_invite = std::max(item.gesture_invite, extra_of(candidate, ATTENTION_EXTRA_GESTURE_INVITE));
    item.gesture_reject = std::max(item.gesture_reject, extra_of(candidate, ATTENTION_EXTRA_GESTURE_REJECT));
    item.expected_answer = std::max(item.expected_answer, unit(candidate.memory_expected_answer));
    item.dialogue_target = std::max(item.dialogue_target, unit(candidate.memory_dialogue_target));
    item.directed_call = std::max(item.directed_call, unit(candidate.lang_directed_call));
    item.question = std::max(item.question, unit(candidate.lang_question));
    item.answer_continuation = std::max(item.answer_continuation, unit(candidate.lang_answer_continuation));
    item.topic_continuation = std::max(item.topic_continuation, unit(candidate.lang_topic_continuation));
    item.addresses_other = std::max(item.addresses_other, unit(candidate.lang_addresses_other));
    item.backchannel = std::max(item.backchannel, unit(candidate.lang_backchannel));
    item.motivation = std::max(item.motivation, unit(candidate.motivation));
    const std::string event = read_id(candidate.event_id, ATTENTION_ID_MAX);
    if (!event.empty()) item.event_id = event;
  }

  /* An unplugged source is erased, not replaced by a neutral guess: an ablation
   * run must really lose that evidence, including from the reported reasons. */
  for (auto &entry : evidence_) {
    PersonEvidence &item = entry.second;
    if (!frame.source_audio_visual) {
      item.gaze = item.body_facing = item.lip_sync = item.sync_onset = 0.0;
      item.gesture_invite = item.gesture_reject = 0.0;
    }
    if (!frame.source_memory_context) {
      item.participant = item.expected_answer = item.dialogue_target = 0.0;
      item.familiarity = item.open_exchange = 0.0;
    }
    if (!frame.source_linguistic_context) {
      item.directed_call = item.question = item.answer_continuation = 0.0;
      item.topic_continuation = item.addresses_other = item.backchannel = 0.0;
    }
    if (!frame.source_internal_state) item.motivation = 0.0;

    /* Looking at us and speaking to us are not alternatives, they compound: the
     * second is only evidence about us to the degree the first holds. Taking the
     * stronger of the two (as an earlier version did) could not separate "just
     * watching" from "watching and talking to me", and left the most ordinary
     * case of all — walking up and speaking — permanently below threshold. */
    const double facing = item.gaze * item.body_facing;
    item.audio_visual = facing * (1.0 + kSpeakingToUs * item.lip_sync * item.voice_activity);
    /* Everything memory knows is attached to an identity, so it is only worth
     * as much as the recognition behind it. A weak face or voiceprint match
     * must not let somebody else's history decide that we are being addressed. */
    item.memory = (0.45 * item.expected_answer + 0.30 * item.dialogue_target + 0.25 * item.participant +
                   kWeightFamiliarity * item.familiarity +
                   kWeightOpenExchange * item.open_exchange) * item.identity;
    item.against = 0.70 * item.addresses_other + 0.30 * item.backchannel;
  }

  /* People who left the frame keep decaying instead of being forgotten at once. */
  for (auto &entry : people_) {
    if (evidence_.find(entry.first) == evidence_.end()) evidence_[entry.first];
  }

  for (auto &entry : evidence_) {
    PersonState &person = people_[entry.first];
    const PersonEvidence &item = entry.second;
    if (!item.candidates.empty()) person.last_seen = frame.stamp_ms;

    /* Somebody who keeps looking and says nothing is asking for something. The
     * previous version let that asymptote below threshold on purpose, so the
     * robot never opened its mouth first; in the field a recognised person
     * looked at it for 115 s and nothing happened. */
    PersonEvidence &mutable_item = entry.second;
    const bool mutual = frame.source_audio_visual && item.gaze >= 0.6 && item.voice_activity < 0.3 &&
                        item.identity >= kEvidenceBit;
    if (mutual) {
      person.gaze_hold_ms += dt * 1000.0;
    } else if (item.gaze < 0.45 || item.voice_activity >= 0.3) {
      person.gaze_hold_ms = 0.0;
      person.gaze_invited_ms = 0;
    }
    const bool answered = person.gaze_invited_ms > 0 &&
                          frame.stamp_ms - person.gaze_invited_ms >= kGazeInviteHoldMs;
    if (!answered && person.gaze_hold_ms > tuned_.gaze_invite_ms) {
      mutable_item.sustained_gaze =
          unit((person.gaze_hold_ms - tuned_.gaze_invite_ms) /
               (tuned_.gaze_saturate_ms - tuned_.gaze_invite_ms)) * item.identity;
    }
    const double coupling = kCouplingFloor + (1.0 - kCouplingFloor) * unit(item.activation);
    const double drive = coupling * (tuned_.weight_audio_visual * item.audio_visual +
                                     tuned_.weight_memory * item.memory +
                                     tuned_.sustained_gaze * mutable_item.sustained_gaze) -
                         tuned_.weight_against * item.against;
    const double equilibrium = tuned_.engage_tau * drive;
    const double step = equilibrium > person.logit ? std::min(dt, kMaxRisingDt) : dt;
    person.logit = equilibrium + (person.logit - equilibrium) * std::exp(-step / tuned_.engage_tau);

    /* Somebody turning to us and starting to speak has to be answerable within a
     * couple of hundred milliseconds. The tonic channel cannot do that at any
     * weight — its time constant is over a second — so the onset enters as an
     * impulse, scaled by how much they were actually facing us. Merely looking
     * produces no onset, and so no impulse. */
    if (frame.source_audio_visual && item.sync_onset >= 0.5) {
      person.logit += kPhasicSyncOnset * item.gaze * item.body_facing * item.lip_sync * item.voice_activity;
    }
    /* Waving at the robot is the visual equivalent of calling it by name:
     * intentional, discrete, and meaningless as a level. Being waved off is
     * weighted higher than being waved at, because getting that one wrong is
     * the more intrusive mistake. */
    if (frame.source_audio_visual && (item.gesture_invite > 0.0 || item.gesture_reject > 0.0)) {
      person.logit += kPhasicInvite * item.gesture_invite - kPhasicReject * item.gesture_reject;
      if (item.gesture_invite > 0.0) person.invited_ms = frame.stamp_ms;
      if (item.gesture_reject > 0.0) person.rejected_ms = frame.stamp_ms;
    }
    person.logit = std::min(kLogitLimit, std::max(-kLogitLimit, person.logit));

    /* Language arrives once per utterance, not once per 50 ms frame. An answer
     * only counts when memory says an answer is owed, which is the acoustic
     * plus lexical conjunction the addressee-detection work relies on, and
     * believing it answers what we asked X means believing the speaker is X.
     * A name call is neither: it works from a voice we have never heard. */
    if (frame.source_linguistic_context && !frame.reflex_active && !item.event_id.empty() &&
        item.event_id != person.last_event) {
      person.last_event = item.event_id;
      person.logit += kPhasicCall * item.directed_call +
                      kPhasicAnswer * item.answer_continuation * item.identity *
                          std::max(item.expected_answer,
                                   std::max(item.dialogue_target, item.open_exchange)) +
                      kPhasicTopic * item.topic_continuation +
                      kPhasicQuestion * item.question * (1.0 - item.addresses_other) -
                      kPhasicOther * item.addresses_other;
    }
    person.logit = std::min(kLogitLimit, std::max(-kLogitLimit, person.logit));
  }
}

inline void EngineBase::report(const AttentionFrameIn &frame) {
  output_.clear();
  output_.reserve(updated_.size());
  std::map<std::string, std::size_t> row_of_candidate;
  for (const auto &entry : updated_) {
    const std::string &key = entry.first;
    const CandidateState &value = entry.second;
    if (frame.stamp_ms - value.last_seen >= 60000) continue;
    AttentionCandidateOut row{};
    copy_id(row.candidate_id, ATTENTION_ID_MAX, key);
    copy_id(row.modality, ATTENTION_MODALITY_MAX, value.modality);
    row.activation = value.activation;
    row.habituation = value.habituation;
    row.inhibition = value.inhibition;
    row.change = value.change;
    row.valid = live_.find(key) != live_.end() ? 1 : 0;
    if (row.valid) {
      const Drive &drive = drives_[key];
      row.input = drive.input;
      row.normalized = drive.normalized;
      row.goal = drive.goal;
      row.uncertainty = drive.uncertainty;
      row.learning = drive.learning;
    }
    row.is_focus = (value.modality == "visual" ? focus_visual_ : focus_audio_) == key ? 1 : 0;
    row_of_candidate[key] = output_.size();
    output_.push_back(row);
  }

  std::string engaged;
  double best = 0.0;
  uint32_t engaged_mask = 0;
  for (const auto &entry : evidence_) {
    const PersonEvidence &item = entry.second;
    const double logit = people_[entry.first].logit;
    const double engagement = sigmoid(logit);
    const double addressee = sigmoid(logit + 1.4 * (frame.source_linguistic_context ? item.directed_call : 0.0));

    uint32_t mask = 0;
    if (item.gaze >= 0.5 && item.audio_visual >= kEvidenceBit) mask |= ATTENTION_REASON_GAZE_ENGAGED;
    if (item.lip_sync * item.voice_activity >= kEvidenceBit) mask |= ATTENTION_REASON_LIP_AUDIO_SYNC;
    if (item.directed_call >= 0.5) mask |= ATTENTION_REASON_DIRECTED_CALL;
    const PersonState &history = people_[entry.first];
    if (history.invited_ms > 0 && frame.stamp_ms - history.invited_ms < kGestureReasonMs) {
      mask |= ATTENTION_REASON_GESTURE_INVITE;
    }
    if (history.rejected_ms > 0 && frame.stamp_ms - history.rejected_ms < kGestureReasonMs) {
      mask |= ATTENTION_REASON_GESTURE_REJECT;
    }
    /* Memory reasons are only reported when the identity they hang on is good
     * enough to have contributed, so the log never claims evidence that was
     * scaled away. */
    const bool identified = item.identity >= kEvidenceBit;
    if (identified && item.participant >= kEvidenceBit) mask |= ATTENTION_REASON_MEMORY_PARTICIPANT;
    if (identified && item.open_exchange >= kEvidenceBit) mask |= ATTENTION_REASON_OPEN_EXCHANGE;
    if (identified && item.familiarity >= kEvidenceBit) mask |= ATTENTION_REASON_FAMILIAR;
    if (identified && item.expected_answer >= 0.5) mask |= ATTENTION_REASON_EXPECTED_ANSWER;
    if (item.topic_continuation >= 0.5 || item.answer_continuation >= 0.5) mask |= ATTENTION_REASON_TOPIC_CONTINUATION;
    if (item.addresses_other >= 0.5) mask |= ATTENTION_REASON_ADDRESSES_OTHER;
    if (item.backchannel >= 0.5) mask |= ATTENTION_REASON_BACKCHANNEL;
    if (identified && item.dialogue_target >= 0.5) mask |= ATTENTION_REASON_DIALOGUE_TARGET;
    if (frame.source_internal_state && item.motivation >= kEvidenceBit) mask |= ATTENTION_REASON_INTERNAL_STATE;
    const bool invitable = item.identity * std::max(item.participant, item.familiarity) >= kEvidenceBit ||
                           item.sustained_gaze >= kEvidenceBit;
    if (engagement >= tuned_.engage_threshold && item.voice_activity < 0.3 && invitable) {
      mask |= ATTENTION_REASON_SILENT_INVITATION;
      if (item.sustained_gaze >= kEvidenceBit && people_[entry.first].gaze_invited_ms == 0) {
        people_[entry.first].gaze_invited_ms = frame.stamp_ms;
      }
    }
    if (item.sustained_gaze >= kEvidenceBit) mask |= ATTENTION_REASON_SUSTAINED_GAZE;

    for (std::size_t index : item.candidates) {
      const std::string key = read_id(frame.candidates[index].candidate_id, ATTENTION_ID_MAX);
      const auto row = row_of_candidate.find(key);
      if (row == row_of_candidate.end()) continue;
      output_[row->second].engagement = engagement;
      output_[row->second].addressee_robot = addressee;
      output_[row->second].reason_mask = mask;
    }

    if (!item.candidates.empty() && engagement > best) {
      best = engagement;
      engaged = entry.first;
      engaged_mask = mask;
    }
  }

  std::string engagement_state = "IDLE";
  if (!engaged.empty()) {
    const PersonEvidence &item = evidence_[engaged];
    if (best >= tuned_.engage_threshold) {
      if (item.voice_activity >= 0.3 || item.directed_call >= 0.5) {
        engagement_state = (item.expected_answer >= 0.5 && item.directed_call < 0.5) ? "EXPECTED_ANSWER" : "ENGAGED";
      } else {
        engagement_state = (item.identity * std::max(item.participant, item.familiarity) >= kEvidenceBit ||
                            item.sustained_gaze >= kEvidenceBit)
                               ? "INVITED"
                               : "ENGAGED";
      }
    } else if (best >= tuned_.observe_threshold) {
      engagement_state = "OBSERVING";
    }
  }
  if (engagement_state == "IDLE") {
    engaged.clear();
    engaged_mask = 0;
  }

  double residual_visual = 1.0;
  double residual_audio = 1.0;
  for (const auto &entry : updated_) {
    if (frame.stamp_ms - entry.second.last_seen >= 60000) continue;
    (entry.second.modality == "visual" ? residual_visual : residual_audio) -= entry.second.activation;
  }

  state_.clear();
  for (const auto &entry : updated_) {
    if (frame.stamp_ms - entry.second.last_seen < 60000) state_[entry.first] = entry.second;
  }
  for (auto iterator = people_.begin(); iterator != people_.end();) {
    iterator = frame.stamp_ms - iterator->second.last_seen >= 180000 ? people_.erase(iterator) : std::next(iterator);
  }

  frame_out_.abi_version = ATTENTION_PLUGIN_ABI_VERSION;
  frame_out_.stamp_ms = frame.stamp_ms;
  frame_out_.candidate_count = static_cast<int32_t>(output_.size());
  frame_out_.candidates = output_.empty() ? nullptr : output_.data();
  copy_id(frame_out_.visual_focus, ATTENTION_ID_MAX, focus_visual_);
  copy_id(frame_out_.audio_focus, ATTENTION_ID_MAX, focus_audio_);
  copy_id(frame_out_.engagement_state, ATTENTION_STATE_MAX, engagement_state);
  copy_id(frame_out_.engaged_person, ATTENTION_ID_MAX, engaged);
  frame_out_.engagement_confidence = engaged.empty() ? 0.0 : best;
  frame_out_.engagement_reason_mask = engaged_mask;
  frame_out_.residual_visual = std::max(0.0, residual_visual);
  frame_out_.residual_audio = std::max(0.0, residual_audio);
  frame_out_.transition_sequence = transitions_;
}

inline const AttentionFrameOut *EngineBase::update(const AttentionFrameIn &frame) {
  if (frame.abi_version != ATTENTION_PLUGIN_ABI_VERSION) return nullptr;
  if (stamp_ms_ >= 0 && frame.stamp_ms <= stamp_ms_) return nullptr;

  const double dt = stamp_ms_ < 0 ? 0.0 : static_cast<double>(frame.stamp_ms - stamp_ms_) / 1000.0;
  stamp_ms_ = frame.stamp_ms;
  const double gain = std::min(2.0, std::max(0.5, std::isfinite(frame.arousal_gain) ? frame.arousal_gain : 1.0));

  live_.clear();
  drives_.clear();
  for (int32_t index = 0; index < frame.candidate_count; ++index) {
    const AttentionCandidateIn &candidate = frame.candidates[index];
    const std::string modality = read_id(candidate.modality, ATTENTION_MODALITY_MAX);
    if (modality != "visual" && modality != "audio") continue;
    const int64_t age = frame.stamp_ms - candidate.stamp_ms;
    if (age < 0 || age >= candidate.ttl_ms) continue;
    live_[read_id(candidate.candidate_id, ATTENTION_ID_MAX)] = static_cast<std::size_t>(index);
  }

  compete(frame, dt, gain);
  select_focus();
  accumulate_engagement(frame, dt);
  report(frame);
  return &frame_out_;
}

}  // namespace attention

#endif  /* ROBOT_ATTENTION_PERCEPTION_ATTENTION_COMMON_HPP */
