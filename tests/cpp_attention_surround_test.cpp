/* Selective Tuning surround for av_memory_language_v2.
 *
 * Two people standing close together should not pull the focus back and forth.
 * Synthetic evidence only; nothing here is a field measurement.
 */
#include "robot_attention_perception/attention_plugin_abi.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const char *label) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", label);
    std::exit(1);
  }
}

struct Person {
  std::string id;
  double salience;
  double azimuth;
  bool has_azimuth = true;
};

/* Runs `frames` ticks and reports how often the visual focus changed. */
struct Outcome {
  std::string focus;
  int switches = 0;
  std::map<std::string, double> activation;  /* final activation per candidate */

  double of(const std::string &id) const {
    const auto found = activation.find(id);
    return found == activation.end() ? 0.0 : found->second;
  }
};

Outcome run(const std::vector<Person> &people, int frames, double jitter) {
  void *handle = attention_plugin_create();
  Outcome outcome;
  std::string previous;
  std::vector<AttentionKeyValue> storage(people.size());

  for (int tick = 0; tick < frames; ++tick) {
    const int64_t stamp = 1000 + tick * 50;
    std::vector<AttentionCandidateIn> live(people.size());
    for (std::size_t index = 0; index < people.size(); ++index) {
      AttentionCandidateIn &candidate = live[index];
      std::memset(&candidate, 0, sizeof(candidate));
      const std::string key = "visual:" + people[index].id;
      std::strncpy(candidate.candidate_id, key.c_str(), ATTENTION_ID_MAX - 1);
      std::strncpy(candidate.modality, "visual", ATTENTION_MODALITY_MAX - 1);
      std::strncpy(candidate.person_id, people[index].id.c_str(), ATTENTION_ID_MAX - 1);
      candidate.stamp_ms = stamp;
      candidate.ttl_ms = 600;
      candidate.observability = 0.9;
      candidate.identity_confidence = 0.8;
      // One-second blocks in which the neighbour is genuinely the more salient
      // of the two. A shorter flicker is smoothed away by the activation time
      // constant and the focus hysteresis, so it would not test anything.
      const double wobble = (index == 0 ? 1.0 : -1.0) * (((tick / 20) % 2) ? -jitter : jitter);
      candidate.salience = people[index].salience + wobble;
      candidate.gaze = candidate.salience;
      candidate.body_facing = 0.9;
      if (people[index].has_azimuth) {
        std::memset(&storage[index], 0, sizeof(AttentionKeyValue));
        std::strncpy(storage[index].key, ATTENTION_EXTRA_AZIMUTH_DEG, ATTENTION_ID_MAX - 1);
        storage[index].value = people[index].azimuth;
        candidate.extra = &storage[index];
        candidate.extra_count = 1;
      }
    }

    AttentionFrameIn frame{};
    frame.abi_version = ATTENTION_PLUGIN_ABI_VERSION;
    frame.stamp_ms = stamp;
    frame.arousal_gain = 1.0;
    frame.source_audio_visual = 1;
    frame.source_memory_context = 1;
    frame.source_linguistic_context = 1;
    frame.candidate_count = static_cast<int32_t>(live.size());
    frame.candidates = live.data();

    const AttentionFrameOut *out = attention_plugin_update(handle, &frame);
    expect(out != nullptr, "frame accepted");
    const std::string focus(out->visual_focus);
    if (tick > 10 && !previous.empty() && focus != previous) ++outcome.switches;
    previous = focus;
    outcome.focus = focus;

    outcome.activation.clear();
    for (int32_t index = 0; index < out->candidate_count; ++index) {
      outcome.activation[out->candidates[index].candidate_id] = out->candidates[index].activation;
    }
  }
  attention_plugin_destroy(handle);
  return outcome;
}

/* Two people side by side, one slightly more salient, with a wobble large
 * enough that the weaker one would otherwise take the focus on every other
 * frame. Compared against the identical scene with no measured bearing, which
 * is the only difference the surround can act on. */
void test_a_close_neighbour_stops_stealing_the_focus() {
  const Outcome unknown = run({{"left", 0.95, 0.0, false}, {"right", 0.60, 12.0, false}}, 120, 0.30);
  const Outcome known = run({{"left", 0.95, 0.0}, {"right", 0.60, 12.0}}, 120, 0.30);
  expect(unknown.switches > 0, "without a bearing the wobble does flip the focus");
  expect(known.switches == 0, "the surround holds the focus through the same wobble");
  expect(known.focus == "visual:left", "the more salient of the pair is the one held");
  expect(known.of("visual:right") < unknown.of("visual:right"),
         "the suppressed neighbour ends with less response than the unsuppressed one");
}

/* Far apart: outside the surround, so nothing is suppressed. */
void test_distant_candidates_are_not_suppressed() {
  const Outcome far = run({{"left", 0.95, -60.0}, {"right", 0.60, 60.0}}, 120, 0.30);
  const Outcome close = run({{"left", 0.95, 0.0}, {"right", 0.60, 12.0}}, 120, 0.30);
  expect(far.of("visual:right") > close.of("visual:right"),
         "a distant candidate keeps more response than a suppressed neighbour");
}

/* Two equally salient candidates leave the focus undecided rather than being
 * arbitrarily separated by the surround. */
void test_an_undecided_pair_is_not_forced_apart() {
  const Outcome tie = run({{"left", 0.80, 0.0}, {"right", 0.80, 12.0}}, 40, 0.0);
  expect(tie.focus.empty(), "an exact tie stays undecided instead of being broken by suppression");
}

/* A clearly stronger neighbour still wins: the surround sharpens selection, it
 * does not lock the focus in place. */
void test_a_stronger_neighbour_can_still_take_over() {
  const Outcome taken = run({{"left", 0.35, 0.0}, {"right", 0.95, 12.0}}, 80, 0.0);
  expect(taken.focus == "visual:right", "the surround must not freeze a weak winner in place");
}

}  // namespace

int main() {
  test_a_close_neighbour_stops_stealing_the_focus();
  test_distant_candidates_are_not_suppressed();
  test_an_undecided_pair_is_not_forced_apart();
  test_a_stronger_neighbour_can_still_take_over();
  std::printf("attention_surround_test ok\n");
  return 0;
}
