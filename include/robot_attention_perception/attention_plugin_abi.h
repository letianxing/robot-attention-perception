/* Stable C ABI for swappable attention algorithms.
 *
 * A plugin receives evidence that perception/memory/internal-state adapters have
 * already normalised, and returns a competition distribution plus a per-person
 * engagement belief. A plugin never gains permission to speak, move, invent an
 * unobserved person or rewrite an identity: the host keeps those decisions.
 *
 * Structs are plain data, fixed layout, no allocation across the boundary.
 * Extending the model means bumping ATTENTION_PLUGIN_ABI_VERSION or adding
 * entries to the generic key/value arrays, never reordering existing fields.
 */
#ifndef ROBOT_ATTENTION_PERCEPTION_ATTENTION_PLUGIN_ABI_H
#define ROBOT_ATTENTION_PERCEPTION_ATTENTION_PLUGIN_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATTENTION_PLUGIN_ABI_VERSION 1
#define ATTENTION_ID_MAX 64
#define ATTENTION_MODALITY_MAX 16
#define ATTENTION_STATE_MAX 32

/* Bits reported per candidate and for the engaged person. They describe which
 * evidence actually moved the belief; they are not a promise that the evidence
 * was correct. */
enum {
  ATTENTION_REASON_GAZE_ENGAGED = 1u << 0,
  ATTENTION_REASON_LIP_AUDIO_SYNC = 1u << 1,
  ATTENTION_REASON_DIRECTED_CALL = 1u << 2,
  ATTENTION_REASON_MEMORY_PARTICIPANT = 1u << 3,
  ATTENTION_REASON_EXPECTED_ANSWER = 1u << 4,
  ATTENTION_REASON_TOPIC_CONTINUATION = 1u << 5,
  ATTENTION_REASON_ADDRESSES_OTHER = 1u << 6,
  ATTENTION_REASON_BACKCHANNEL = 1u << 7,
  ATTENTION_REASON_SILENT_INVITATION = 1u << 8,
  ATTENTION_REASON_DIALOGUE_TARGET = 1u << 9,
  ATTENTION_REASON_INTERNAL_STATE = 1u << 10,
  ATTENTION_REASON_SOUND_NOVELTY = 1u << 11,
  ATTENTION_REASON_FAMILIAR = 1u << 12,
  ATTENTION_REASON_GESTURE_INVITE = 1u << 13,
  ATTENTION_REASON_GESTURE_REJECT = 1u << 14,
  ATTENTION_REASON_OPEN_EXCHANGE = 1u << 15,
  ATTENTION_REASON_SUSTAINED_GAZE = 1u << 16
};

/* Keys carried in AttentionCandidateIn.extra. Adding one here does not change
 * any struct layout, so an already-compiled plugin keeps working and simply
 * ignores the key it does not know. */
#define ATTENTION_EXTRA_MEMORY_FAMILIARITY "memory_familiarity"
/* Bearing in degrees, needed by algorithms with a spatial inhibitory
 * surround. Absent when the host could not measure a direction. */
#define ATTENTION_EXTRA_AZIMUTH_DEG "azimuth_deg"
/* 1.0 on the single frame where lip-audio synchrony starts while the person
 * is facing the robot. The onset of audiovisual synchrony is an event, not a
 * level: a tonic channel cannot answer within a a few hundred milliseconds,
 * so this is the phasic counterpart of the language impulses. The host owns
 * firing it exactly once per episode. */
#define ATTENTION_EXTRA_AV_SYNC_ONSET "av_sync_onset"
/* A beckoning or waving gesture aimed at the robot, and a warding-off one,
 * scored 0..1 on the frame the gesture is first recognised. A wave is the
 * visual equivalent of being called by name: intentional, discrete, and
 * meaningless as a level, so it arrives as an onset like the others. */
#define ATTENTION_EXTRA_GESTURE_INVITE "gesture_invite"
#define ATTENTION_EXTRA_GESTURE_REJECT "gesture_reject"
/* 1.0 on the frame a person becomes visible again after being away. Coming
 * back is a change, so it lifts that person's habituation. */
#define ATTENTION_EXTRA_RETURNED "returned_after_absence"
/* How open an exchange with this person still is, decayed from when they last
 * addressed the robot. Losing sight of somebody does not end a conversation
 * that already started. */
#define ATTENTION_EXTRA_OPEN_EXCHANGE "memory_open_exchange"

typedef struct {
  char key[ATTENTION_ID_MAX];
  double value;
} AttentionKeyValue;

typedef struct {
  char candidate_id[ATTENTION_ID_MAX]; /* "visual:<person>" / "audio:<track>" */
  char modality[ATTENTION_MODALITY_MAX]; /* "visual" or "audio" */
  char person_id[ATTENTION_ID_MAX];      /* empty when not resolved */

  int64_t stamp_ms;
  int32_t ttl_ms;

  /* audio_visual source: measured now, already freshness-checked by the host. */
  double salience;
  double goal;
  double surprise;
  double observability;
  double uncertainty;
  double importance; /* stable prior such as a registered owner role */
  double gaze;
  double body_facing;
  double lip_sync;
  double voice_activity;
  double identity_confidence;

  /* memory_context source: working-memory facts about this same session. */
  double memory_participant;      /* took part in the ongoing episode */
  double memory_recency;          /* decayed time since they last spoke */
  double memory_expected_answer;  /* robot asked them something still open */
  double memory_dialogue_target;  /* currently exchanging turns with robot */

  /* linguistic_context source: features of the utterance in progress. */
  double lang_directed_call;
  double lang_question;
  double lang_answer_continuation;
  double lang_topic_continuation;
  double lang_addresses_other;
  double lang_backchannel;

  /* internal_state source: bounded modulation only, never a fabricated value. */
  double motivation;

  /* Cross-modal association produced by the host binder. */
  char link_id[ATTENTION_ID_MAX];
  double link_confidence;

  /* Identifiers used to apply an effect exactly once. */
  char change_id[ATTENTION_ID_MAX]; /* acoustic novelty onset */
  char event_id[ATTENTION_ID_MAX];  /* utterance boundary for phasic language */

  int32_t extra_count;
  const AttentionKeyValue *extra;
} AttentionCandidateIn;

typedef struct {
  int32_t abi_version;
  int64_t stamp_ms;
  double arousal_gain; /* 1.0 means "no modulation", not "measured calm" */

  /* Which adapters the operator has plugged in for this frame. A disabled
   * source contributes nothing; it is not replaced by a neutral guess. */
  int32_t source_audio_visual;
  int32_t source_memory_context;
  int32_t source_linguistic_context;
  int32_t source_internal_state;

  int32_t robot_speaking;
  int32_t reflex_active;

  int32_t candidate_count;
  const AttentionCandidateIn *candidates;
  int32_t extra_count;
  const AttentionKeyValue *extra;
} AttentionFrameIn;

typedef struct {
  char candidate_id[ATTENTION_ID_MAX];
  char modality[ATTENTION_MODALITY_MAX];
  double activation;
  double habituation;
  double inhibition;
  double change;
  double normalized;
  double input;
  double goal;        /* top-down term the algorithm actually used */
  double uncertainty; /* echoed so the host can rank information value */
  double learning;
  double engagement;      /* persistent belief that they are engaging the robot */
  double addressee_robot; /* belief that the current utterance addresses it */
  int32_t valid;
  int32_t is_focus;
  uint32_t reason_mask;
} AttentionCandidateOut;

typedef struct {
  int32_t abi_version;
  int64_t stamp_ms;

  int32_t candidate_count;
  const AttentionCandidateOut *candidates;

  char visual_focus[ATTENTION_ID_MAX];
  char audio_focus[ATTENTION_ID_MAX];

  /* IDLE / OBSERVING / ENGAGED / INVITED / EXPECTED_ANSWER */
  char engagement_state[ATTENTION_STATE_MAX];
  char engaged_person[ATTENTION_ID_MAX];
  double engagement_confidence;
  uint32_t engagement_reason_mask;

  double residual_visual;
  double residual_audio;
  int32_t transition_sequence;
} AttentionFrameOut;

enum {
  ATTENTION_STRUCT_FRAME_IN = 0,
  ATTENTION_STRUCT_CANDIDATE_IN = 1,
  ATTENTION_STRUCT_FRAME_OUT = 2,
  ATTENTION_STRUCT_CANDIDATE_OUT = 3,
  ATTENTION_STRUCT_KEY_VALUE = 4
};

/* Must equal ATTENTION_PLUGIN_ABI_VERSION or the host refuses to load. */
int32_t attention_plugin_abi_version(void);

/* sizeof() of the structs above, so a host written in another language can
 * verify its own layout instead of silently reading garbage. */
int32_t attention_plugin_struct_size(int32_t which);

/* Static JSON manifest: id, display_name, version, required/optional sources,
 * calibrated flag and references. Valid for the lifetime of the library. */
const char *attention_plugin_manifest(void);

void *attention_plugin_create(void);
void attention_plugin_destroy(void *handle);

/* Returns a buffer owned by the handle, valid until the next update or destroy.
 * Returns NULL when the frame is rejected (bad ABI or non-monotonic stamp). */
const AttentionFrameOut *attention_plugin_update(void *handle,
                                                 const AttentionFrameIn *frame);

#ifdef __cplusplus
}
#endif

#endif /* ROBOT_ATTENTION_PERCEPTION_ATTENTION_PLUGIN_ABI_H */
