#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from hri_msgs.msg import EngagementLevel, IdsList, LiveSpeech
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, String
    from tf2_ros import TransformBroadcaster

    class NativeIngress(Node):
        def __init__(self):
            super().__init__("mac_native_perception_ingress")
            self.tf = TransformBroadcaster(self)
            self._dynamic_publishers = {}
            self.face_ids = self.create_publisher(IdsList, "/humans/faces/tracked", 10)
            self.body_ids = self.create_publisher(IdsList, "/humans/bodies/tracked", 10)
            self.voice_ids = self.create_publisher(IdsList, "/humans/voices/tracked", 10)
            self.person_ids = self.create_publisher(IdsList, "/humans/persons/tracked", 10)
            self.create_subscription(String, "/vision/people_json", self.on_vision, 10)
            self.create_subscription(String, "/voice/acoustic_tracks", self.on_voice, 10)

        def publisher(self, message_type, topic):
            key = (message_type, topic)
            if key not in self._dynamic_publishers:
                self._dynamic_publishers[key] = self.create_publisher(message_type, topic, 10)
            return self._dynamic_publishers[key]

        def ids(self, publisher, values):
            message = IdsList()
            message.header.stamp = self.get_clock().now().to_msg()
            message.ids = list(values)
            publisher.publish(message)

        def on_vision(self, message):
            try:
                people = json.loads(message.data).get("people") or []
                faces = [str(item["face_id"]) for item in people if item.get("face_id")]
                bodies = [str(item["body_id"]) for item in people if item.get("body_id")]
                persons = [str(item["person_id"]) for item in people if item.get("person_id")]
                self.ids(self.face_ids, faces)
                self.ids(self.body_ids, bodies)
                self.ids(self.person_ids, persons)
                for item in people:
                    self.publish_person(item)
                    azimuth = item.get("azimuth_deg")
                    elevation = item.get("elevation_deg")
                    if azimuth is None:
                        continue
                    if item.get("face_id"):
                        self.send_direction_frame(f"face_{item['face_id']}", azimuth, elevation, item.get("distance_m"))
                        self.send_direction_frame(f"gaze_{item['face_id']}", azimuth, elevation, item.get("distance_m"))
                    if item.get("body_id"):
                        self.send_direction_frame(f"body_{item['body_id']}", azimuth, elevation, item.get("distance_m"))
            except Exception as exc:
                self.get_logger().warning(f"invalid vision ingress: {exc}")

        def publish_person(self, item):
            person_id = str(item.get("person_id") or "")
            if not person_id:
                return
            stamp = self.get_clock().now().to_msg()
            anonymous = Bool(data=str(item.get("role") or "unknown") not in {"owner", "known"})
            self.publisher(Bool, f"/humans/persons/{person_id}/anonymous").publish(anonymous)
            for field in ("face_id", "body_id", "voice_id"):
                value = String(data=str(item.get(field) or ""))
                self.publisher(String, f"/humans/persons/{person_id}/{field}").publish(value)
            confidence = Float32(data=float(item.get("face_confidence", 0.0) or 0.0))
            self.publisher(Float32, f"/humans/persons/{person_id}/location_confidence").publish(confidence)
            engagement = EngagementLevel()
            engagement.header.stamp = stamp
            levels = {
                "disengaged": EngagementLevel.DISENGAGED,
                "unengaged": EngagementLevel.DISENGAGED,
                "engaging": EngagementLevel.ENGAGING,
                "engaged": EngagementLevel.ENGAGED,
                "disengaging": EngagementLevel.DISENGAGING,
            }
            engagement.level = levels.get(str(item.get("engagement_status") or "unknown"), EngagementLevel.UNKNOWN)
            self.publisher(EngagementLevel, f"/humans/persons/{person_id}/engagement_status").publish(engagement)

        def on_voice(self, message):
            try:
                tracks = [
                    item for item in (json.loads(message.data).get("tracks") or [])
                    if str(item.get("track_id") or "").lower() not in {"voice_demo", "demo_voice"}
                ]
                ids = [str(item["track_id"]) for item in tracks if item.get("track_id")]
                self.ids(self.voice_ids, ids)
                for item in tracks:
                    voice_id = str(item.get("track_id") or "")
                    if not voice_id:
                        continue
                    self.publisher(Bool, f"/humans/voices/{voice_id}/is_speaking").publish(
                        Bool(data=bool(item.get("voice_activity", False)))
                    )
                    transcript = item.get("transcript")
                    if isinstance(transcript, dict):
                        speech = LiveSpeech()
                        speech.header.stamp = self.get_clock().now().to_msg()
                        speech.incremental = "" if transcript.get("is_final") else str(transcript.get("text") or "")
                        speech.final = str(transcript.get("text") or "") if transcript.get("is_final") else ""
                        speech.confidence = float(transcript.get("confidence", 0.0) or 0.0)
                        speech.locale = str(transcript.get("language") or "")
                        self.publisher(LiveSpeech, f"/humans/voices/{voice_id}/speech").publish(speech)
                    if item.get("azimuth_deg") is not None:
                        self.send_direction_frame(
                            f"voice_{voice_id}", item["azimuth_deg"], item.get("elevation_deg"), item.get("distance_m")
                        )
            except Exception as exc:
                self.get_logger().warning(f"invalid voice ingress: {exc}")

        def send_direction_frame(self, child, azimuth_deg, elevation_deg, distance_m):
            azimuth = math.radians(float(azimuth_deg or 0.0))
            elevation = math.radians(float(elevation_deg or 0.0))
            distance = max(0.1, float(distance_m or 1.0))
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "base_link"
            transform.child_frame_id = child
            transform.transform.translation.x = distance * math.cos(elevation) * math.cos(azimuth)
            transform.transform.translation.y = distance * math.cos(elevation) * math.sin(azimuth)
            transform.transform.translation.z = distance * math.sin(elevation)
            transform.transform.rotation.z = math.sin(azimuth * 0.5)
            transform.transform.rotation.w = math.cos(azimuth * 0.5)
            self.tf.sendTransform(transform)

    rclpy.init()
    node = NativeIngress()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
