#!/usr/bin/env python3
"""Adapt the official pyhri HRIListener view to the typed attention input.

This keeps the attention algorithm independent from detector implementations:
official ROS4HRI nodes (or our ingress bridge) publish /humans/*, pyhri reads
those topics, and this adapter emits the existing typed People contract.
"""
import math
import time

import rclpy
from rclpy.node import Node
from perception_interfaces.msg import People, VisionPerson


def _set(msg, name, value):
    if hasattr(msg, name):
        setattr(msg, name, value)


def _bearing(feature):
    transform = getattr(feature, "transform", None)
    if transform is None:
        return None, None, None
    t = transform.transform.translation
    distance = math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)
    return math.degrees(math.atan2(t.y, t.x)), math.degrees(math.atan2(t.z, max(math.hypot(t.x, t.y), 1e-6))), distance


class PyHriPeopleAdapter(Node):
    def __init__(self):
        super().__init__("pyhri_people_adapter")
        try:
            import hri
        except ImportError as exc:
            raise RuntimeError("pyhri (import hri) is required") from exc
        self.listener = hri.HRIListener.create(self)
        self.listener.set_reference_frame(self.declare_parameter("reference_frame", "base_link").value)
        topic = self.declare_parameter("people_topic", "/perception/vision/people").value
        self.pub = self.create_publisher(People, topic, 10)
        self.create_timer(0.1, self.publish_people)

    def publish_people(self):
        message = People()
        message.header.stamp = self.get_clock().now().to_msg()
        for person in self.listener.tracked_persons.values():
            item = VisionPerson()
            _set(item, "person_id", person.id)
            face = person.face
            body = person.body
            voice = person.voice
            _set(item, "face_id", getattr(face, "id", "") if face else "")
            _set(item, "body_id", getattr(body, "id", "") if body else "")
            _set(item, "voice_id", getattr(voice, "id", "") if voice else "")
            _set(item, "role", "known" if not person.anonymous else "anonymous")
            source = face or body or voice
            azimuth, elevation, distance = _bearing(source) if source else (None, None, None)
            if azimuth is not None:
                _set(item, "has_azimuth", True); _set(item, "azimuth_deg", azimuth)
                _set(item, "has_elevation", True); _set(item, "elevation_deg", elevation)
                _set(item, "has_distance", True); _set(item, "distance_m", distance)
            engagement = getattr(person, "engagement_status", None)
            _set(item, "engagement_status", getattr(engagement, "name", "unknown").lower() if engagement else "unknown")
            _set(item, "identity_confidence", 1.0 if not person.anonymous else 0.0)
            if face:
                expression = getattr(face, "expression_va", None)
                if expression:
                    _set(item, "emotion_valence", float(expression.valence)); _set(item, "emotion_arousal", float(expression.arousal)); _set(item, "emotion_valid", True)
                    _set(item, "emotion_label", getattr(getattr(face, "expression", None), "name", "unknown").lower())
                _set(item, "face_visible", True)
                _set(item, "face_confidence", float(getattr(face, "expression_confidence", 0.0) or 0.0))
            message.people.append(item)
        self.pub.publish(message)


def main():
    rclpy.init()
    node = PyHriPeopleAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
