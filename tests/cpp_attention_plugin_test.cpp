/* Behavioural checks for av_memory_language_v1.
 *
 * These assert that the documented interaction cases fall out of one fused
 * update instead of scripted branches. They exercise the algorithm on
 * synthetic evidence only and say nothing about field accuracy.
 */
#include "robot_attention_perception/attention_plugin_abi.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

AttentionCandidateIn make(const std::string &id, const std::string &modality, const std::string &person,
                          int64_t stamp_ms) {
  AttentionCandidateIn candidate{};
  std::strncpy(candidate.candidate_id, id.c_str(), ATTENTION_ID_MAX - 1);
  std::strncpy(candidate.modality, modality.c_str(), ATTENTION_MODALITY_MAX - 1);
  std::strncpy(candidate.person_id, person.c_str(), ATTENTION_ID_MAX - 1);
  candidate.stamp_ms = stamp_ms;
  candidate.ttl_ms = modality == "audio" ? 350 : 600;
  candidate.observability = 0.9;
  return candidate;
}

struct Run {
  std::string state;
  std::string person;
  double confidence = 0.0;
  uint32_t reasons = 0;
  /* An invitation is a state that comes and goes inside a run, so the last
   * frame alone cannot tell us whether it ever happened. */
  bool saw_invited = false;
  uint32_t invited_reasons = 0;
};

/* Feeds `frames` ticks at 50 ms and returns the final decision. */
Run drive(void *handle, int frames, int64_t start_ms,
          const std::vector<AttentionCandidateIn> &candidates, bool memory_source = true,
          bool language_source = true) {
  Run result;
  for (int tick = 0; tick < frames; ++tick) {
    const int64_t stamp = start_ms + tick * 50;
    std::vector<AttentionCandidateIn> live = candidates;
    for (auto &candidate : live) candidate.stamp_ms = stamp;

    AttentionFrameIn frame{};
    frame.abi_version = ATTENTION_PLUGIN_ABI_VERSION;
    frame.stamp_ms = stamp;
    frame.arousal_gain = 1.0;
    frame.source_audio_visual = 1;
    frame.source_memory_context = memory_source ? 1 : 0;
    frame.source_linguistic_context = language_source ? 1 : 0;
    frame.source_internal_state = 0;
    frame.candidate_count = static_cast<int32_t>(live.size());
    frame.candidates = live.data();

    const AttentionFrameOut *out = attention_plugin_update(handle, &frame);
    assert(out != nullptr);
    result.state = std::string(out->engagement_state);
    result.person = std::string(out->engaged_person);
    result.confidence = out->engagement_confidence;
    result.reasons = out->engagement_reason_mask;
    if (result.state == "INVITED") {
      result.saw_invited = true;
      result.invited_reasons |= out->engagement_reason_mask;
    }
  }
  return result;
}

void expect(bool condition, const char *label) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", label);
    std::exit(1);
  }
}

/* A glance is interest; an unbroken stare is a request. The robot used to sit
 * silent through both, which in the field meant a recognised person looking at
 * it for nearly two minutes and nothing happening. */
void test_a_glance_is_not_an_invitation_but_a_long_stare_is() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn person = make("visual:guest", "visual", "guest", 0);
  person.salience = 0.85;
  person.goal = 0.8;
  person.gaze = 0.9;
  person.body_facing = 0.9;
  person.identity_confidence = 0.7;
  const Run glance = drive(handle, 40, 1000, {person});  /* 2 s */
  expect(glance.state != "INVITED", "two seconds of looking is not an invitation");
  expect(glance.confidence < 0.72, "a glance stays below the engagement threshold");

  const Run stare = drive(handle, 160, 3000, {person});  /* 8 s more, unbroken */
  expect(stare.saw_invited, "an unbroken silent stare eventually invites");
  expect((stare.invited_reasons & ATTENTION_REASON_SUSTAINED_GAZE) != 0, "sustained gaze reported as the reason");
  expect((stare.invited_reasons & ATTENTION_REASON_SILENT_INVITATION) != 0, "invitation reason reported");

  /* Answered once. Someone who simply works in front of the robot is not asked
   * again until they look away and look back. */
  const Run after = drive(handle, 200, 11000, {person});
  expect(!after.saw_invited, "one invitation per episode of gaze");
  attention_plugin_destroy(handle);
}

/* An unrecognised, flickering detection cannot stare at anybody. */
void test_a_stare_needs_a_recognised_person() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn person = make("visual:blur", "visual", "blur", 0);
  person.salience = 0.85;
  person.goal = 0.8;
  person.gaze = 0.9;
  person.body_facing = 0.9;
  person.identity_confidence = 0.05;
  const Run result = drive(handle, 240, 1000, {person});
  expect(!result.saw_invited, "no invitation without a person to invite");
  attention_plugin_destroy(handle);
}

/* Two people were talking; one of them turns to the robot and stays silent. */
void test_participant_gaze_becomes_invitation() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn looker = make("visual:owner", "visual", "owner", 0);
  looker.salience = 0.85;
  looker.goal = 0.8;
  looker.gaze = 0.9;
  looker.body_facing = 0.9;
  looker.identity_confidence = 0.8;
  looker.memory_participant = 1.0;
  looker.memory_recency = 0.8;

  const Run early = drive(handle, 10, 1000, {looker});
  expect(early.state != "INVITED", "invitation needs sustained gaze, not one frame");
  const Run later = drive(handle, 40, 1600, {looker});
  expect(later.state == "INVITED", "participant + sustained silent gaze invites");
  expect(later.person == "owner", "the invitation names the person who looked");
  expect((later.reasons & ATTENTION_REASON_SILENT_INVITATION) != 0, "invitation reason reported");
  expect((later.reasons & ATTENTION_REASON_MEMORY_PARTICIPANT) != 0, "memory channel reported");
  attention_plugin_destroy(handle);
}

/* Same evidence with the memory source unplugged must not invite: this is the
 * ablation the console offers, not a cosmetic switch. */
void test_invitation_needs_the_memory_source() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn looker = make("visual:owner", "visual", "owner", 0);
  looker.salience = 0.85;
  looker.goal = 0.8;
  looker.gaze = 0.9;
  looker.body_facing = 0.9;
  looker.identity_confidence = 0.8;
  looker.memory_participant = 1.0;
  looker.memory_recency = 0.8;
  const Run result = drive(handle, 80, 1000, {looker}, /*memory_source=*/false);
  expect(result.state != "INVITED", "memory source disabled removes the invitation");
  attention_plugin_destroy(handle);
}

/* The robot asked a question; the person answers with their head turned away,
 * so there is no gaze and no visible face at all. */
void test_answer_with_head_turned_away() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn voice = make("audio:track-7", "audio", "owner", 0);
  voice.salience = 0.8;
  voice.voice_activity = 1.0;
  voice.identity_confidence = 0.9;  /* a reliable voiceprint match */
  voice.memory_expected_answer = 1.0;
  voice.lang_answer_continuation = 1.0;
  std::strncpy(voice.event_id, "utt-91", ATTENTION_ID_MAX - 1);

  const Run result = drive(handle, 6, 1000, {voice});
  expect(result.state == "EXPECTED_ANSWER", "answer to an open question is accepted without gaze");
  expect(result.person == "owner", "answer is attributed to the asked person");
  expect((result.reasons & ATTENTION_REASON_EXPECTED_ANSWER) != 0, "expected-answer reason reported");
  expect((result.reasons & ATTENTION_REASON_GAZE_ENGAGED) == 0, "no gaze evidence is claimed");
  attention_plugin_destroy(handle);
}

/* The most ordinary case there is: somebody walks up, looks at the robot and
 * speaks. It has to be answerable within a couple of hundred milliseconds, and
 * it must not need a name, a shared history or anything remembered. */
void test_walking_up_and_speaking_engages_quickly() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn face = make("visual:guest", "visual", "guest", 0);
  face.salience = 0.85;
  face.goal = 0.8;
  face.gaze = 0.9;
  face.body_facing = 0.9;
  face.lip_sync = 1.0;
  face.identity_confidence = 0.8;
  AttentionKeyValue onset{};
  std::strncpy(onset.key, ATTENTION_EXTRA_AV_SYNC_ONSET, ATTENTION_ID_MAX - 1);
  onset.value = 1.0;
  face.extra = &onset;
  face.extra_count = 1;

  AttentionCandidateIn voice = make("audio:track-4", "audio", "guest", 0);
  voice.salience = 0.8;
  voice.voice_activity = 1.0;
  voice.identity_confidence = 0.8;

  const Run early = drive(handle, 3, 1000, {face, voice});
  expect(early.state == "ENGAGED", "walking up and speaking must engage within ~150 ms");
  expect(early.person == "guest", "no prior history is needed to be spoken to");
  const Run held = drive(handle, 40, 1200, {face, voice});
  expect(held.state == "ENGAGED", "it must stay engaged while they keep talking to us");
  attention_plugin_destroy(handle);
}

/* Speech alone is not evidence about us: a bystander talking while turned away
 * carries the same lip-audio synchrony and must stay out. */
void test_speech_while_turned_away_is_not_about_us() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn face = make("visual:guest", "visual", "guest", 0);
  face.salience = 0.5;
  face.gaze = 0.1;
  face.body_facing = 0.3;
  face.lip_sync = 1.0;
  face.identity_confidence = 0.8;
  AttentionCandidateIn voice = make("audio:track-5", "audio", "guest", 0);
  voice.salience = 0.8;
  voice.voice_activity = 1.0;
  voice.identity_confidence = 0.8;

  const Run result = drive(handle, 60, 1000, {face, voice});
  expect(result.state != "ENGAGED", "lip-audio synchrony while turned away is not engagement");
  attention_plugin_destroy(handle);
}

/* Memory is attached to an identity, so a weak voiceprint match must not let a
 * question we asked somebody else be answered by whoever happens to speak. */
void test_weak_identity_does_not_inherit_the_expectation() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn voice = make("audio:track-8", "audio", "owner", 0);
  voice.salience = 0.8;
  voice.voice_activity = 1.0;
  voice.identity_confidence = 0.1;  /* barely above the matching floor */
  voice.memory_expected_answer = 1.0;
  voice.lang_answer_continuation = 1.0;
  std::strncpy(voice.event_id, "utt-92", ATTENTION_ID_MAX - 1);

  const Run result = drive(handle, 6, 1000, {voice});
  expect(result.state != "EXPECTED_ANSWER", "an unreliable identity cannot claim the answer");
  attention_plugin_destroy(handle);
}

/* Speaking to another human in the room pushes the belief down, even while the
 * speaker happens to be facing the robot. */
void test_addressing_another_human_suppresses() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn speaker = make("audio:track-3", "audio", "guest", 0);
  speaker.salience = 0.8;
  speaker.voice_activity = 1.0;
  speaker.identity_confidence = 0.9;
  speaker.memory_participant = 1.0;
  speaker.memory_recency = 1.0;
  speaker.lang_addresses_other = 1.0;
  std::strncpy(speaker.event_id, "utt-12", ATTENTION_ID_MAX - 1);

  const Run result = drive(handle, 20, 1000, {speaker});
  expect(result.state != "ENGAGED" && result.state != "EXPECTED_ANSWER",
         "naming another human must not address the robot");
  attention_plugin_destroy(handle);
}

/* Calling the robot by name works off camera and at ordinary volume. */
void test_directed_call_without_a_face() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn voice = make("audio:track-1", "audio", "owner", 0);
  voice.salience = 0.7;
  voice.goal = 1.0;
  voice.voice_activity = 1.0;
  voice.lang_directed_call = 1.0;
  std::strncpy(voice.event_id, "utt-1", ATTENTION_ID_MAX - 1);

  const Run result = drive(handle, 3, 1000, {voice});
  expect(result.state == "ENGAGED", "a name call engages immediately");
  expect((result.reasons & ATTENTION_REASON_DIRECTED_CALL) != 0, "directed call reason reported");
  attention_plugin_destroy(handle);
}

/* Self-talk stays recorded but unanswered until the speaker looks over. */
void test_self_talk_then_gaze() {
  void *handle = attention_plugin_create();
  AttentionCandidateIn voice = make("audio:track-2", "audio", "owner", 0);
  voice.salience = 0.8;
  voice.voice_activity = 1.0;
  std::strncpy(voice.event_id, "utt-5", ATTENTION_ID_MAX - 1);
  AttentionCandidateIn face = make("visual:owner", "visual", "owner", 0);
  face.salience = 0.5;
  face.gaze = 0.1;
  face.body_facing = 0.3;

  const Run talking = drive(handle, 40, 1000, {voice, face});
  expect(talking.state != "ENGAGED", "self-talk alone is recorded, not answered");

  AttentionCandidateIn looking = face;
  looking.salience = 0.85;
  looking.goal = 0.8;
  looking.gaze = 0.9;
  looking.body_facing = 0.9;
  looking.identity_confidence = 0.8;
  looking.memory_participant = 1.0;
  looking.memory_recency = 1.0;
  const Run invited = drive(handle, 50, 3200, {looking});
  expect(invited.state == "INVITED", "turning to the robot after self-talk invites a reply");
  attention_plugin_destroy(handle);
}

void test_manifest_and_abi() {
  expect(attention_plugin_abi_version() == ATTENTION_PLUGIN_ABI_VERSION, "abi version matches");
  const std::string manifest = attention_plugin_manifest();
  // Every algorithm runs the same checks, so the id is read rather than assumed.
  expect(manifest.find("\"id\":\"av_memory_language_v") != std::string::npos, "manifest carries the id");
  expect(manifest.find("\"calibrated\":false") != std::string::npos, "manifest does not claim calibration");
  expect(manifest.find("\"engagement_state\"") != std::string::npos, "manifest declares the engagement output");

  void *handle = attention_plugin_create();
  AttentionFrameIn frame{};
  frame.abi_version = ATTENTION_PLUGIN_ABI_VERSION + 1;
  frame.stamp_ms = 1000;
  expect(attention_plugin_update(handle, &frame) == nullptr, "wrong abi version is rejected");
  frame.abi_version = ATTENTION_PLUGIN_ABI_VERSION;
  expect(attention_plugin_update(handle, &frame) != nullptr, "empty frame is accepted");
  expect(attention_plugin_update(handle, &frame) == nullptr, "a repeated stamp is rejected");
  attention_plugin_destroy(handle);
}

}  // namespace

int main() {
  test_manifest_and_abi();
  test_a_glance_is_not_an_invitation_but_a_long_stare_is();
  test_a_stare_needs_a_recognised_person();
  test_participant_gaze_becomes_invitation();
  test_invitation_needs_the_memory_source();
  test_answer_with_head_turned_away();
  test_walking_up_and_speaking_engages_quickly();
  test_speech_while_turned_away_is_not_about_us();
  test_weak_identity_does_not_inherit_the_expectation();
  test_addressing_another_human_suppresses();
  test_directed_call_without_a_face();
  test_self_talk_then_gaze();
  const std::string manifest = attention_plugin_manifest();
  const std::size_t start = manifest.find("\"id\":\"") + 6;
  std::printf("attention_plugin_test ok (%s)\n", manifest.substr(start, manifest.find('"', start) - start).c_str());
  return 0;
}
