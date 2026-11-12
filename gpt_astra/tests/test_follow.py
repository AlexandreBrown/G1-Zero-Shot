from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpt_astra import calibration
from gpt_astra.calibration import SIM
from gpt_astra.follow import FINE, Follower, approach_x, coverage_gaps, locate, nearest_grab
from gpt_astra.run import follow_problems

# Follower tests use fixed walking numbers, not whatever the measurement tools last wrote into sim.json.
TEST = replace(SIM, coast={0: .25, 1: .1, 2: .06, 3: .06, 4: .08, 5: .02, 6: .06, 7: .06}, settle_back=.06,
               settle_fast=1., settle_fine=.6, stall=2.5, kick=.4, min_step=.05, fine_within=.3, lost_walk=4.,
               max_walk=8., fence=3., walk_speed=.4, fine_walk_speed=.2)


def jpeg(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def balloon_image(point, cal=SIM):
    """The ego image of a red sphere centred at point (pelvis frame) under the calibration's camera."""
    camera = cal.camera_rotation.T @ (point - cal.camera_position)
    focal = 240 / np.tan(np.radians(cal.fovy_deg) / 2)
    u, v = 320 + focal * camera[0] / -camera[2], 240 - focal * camera[1] / -camera[2]
    radius = focal * cal.balloon_radius / -camera[2]
    rows, cols = np.mgrid[:480, :640]
    image = np.full((480, 640, 3), (40, 60, 120), np.uint8)
    image[(cols - u) ** 2 + (rows - v) ** 2 <= radius ** 2] = (220, 10, 10)
    return image


class FollowTests(unittest.TestCase):
    def test_locate_inverts_the_camera_projection_even_when_the_edge_cuts_the_balloon(self):
        # (.6, .1) is fully in view; (.2, 0) is so close that the bottom edge cuts off more than half of it.
        for point in (np.array([.6, .1, .15]), np.array([.2, 0., .13])):
            found = locate(jpeg(balloon_image(point)))
            self.assertTrue(np.allclose(found, point, atol=.015), (point, found))
        self.assertIsNone(locate(jpeg(np.full((480, 640, 3), 90, np.uint8))))

    def test_locate_ignores_red_things_that_are_not_a_held_balloon(self):
        for rows, cols in ((slice(150, 420), slice(200, 330)), (slice(150, 350), slice(200, 400))):
            shirt = np.full((480, 640, 3), (40, 60, 120), np.uint8)
            shirt[rows, cols] = (220, 10, 10)                         # a red rectangle or square: not a circle
            self.assertIsNone(locate(jpeg(shirt)))
        highlight = balloon_image(np.array([.6, .1, .15]))
        highlight[150:400, 300:340][(highlight[150:400, 300:340, 0] == 220)] = (250, 250, 250)
        self.assertTrue(np.allclose(locate(jpeg(highlight)), [.6, .1, .15], atol=.015))   # a white highlight
        self.assertIsNone(locate(jpeg(balloon_image(np.array([1.2, 0., .45])))))   # far above hand height

    def test_a_balloon_and_a_red_label_are_not_merged_into_one_close_balloon(self):
        # Robot run 2026-11-04: with the circle test loosened for a corded balloon, the balloon plus red labels on
        # equipment fitted one huge, mostly empty circle that read as a balloon 0.25 m away; the robot walked back.
        loose = replace(SIM, max_fit_residual=.45)
        image = balloon_image(np.array([1.2, 0., .1]))
        image[400:440, 20:60] = (220, 10, 10)                                 # a red label at the bottom left
        found = locate(jpeg(image), loose)
        self.assertTrue(found is None or found[0] > 1., found)
        self.assertTrue(np.allclose(locate(jpeg(balloon_image(np.array([1.2, 0., .1]))), loose), [1.2, 0., .1],
                                    atol=.03))

    def test_every_point_the_follower_accepts_has_a_grab_pose(self):
        test = replace(TEST, meeting_x=(.42, .45, .47), yaw_tracking=.9)
        self.assertEqual(nearest_grab(np.array([.45, 0., .16]), test), "2_10")   # the longest reach first
        self.assertEqual(nearest_grab(np.array([.42, 0., .16]), test), "1_10")
        self.assertIsNone(nearest_grab(np.array([.7, 0., .16]), test))
        # A property of every calibration file in use: the approach zone is covered by the grab poses.
        for cal in (test, SIM, calibration.load(calibration.for_backend("robot"))):
            self.assertEqual(coverage_gaps(cal), [], cal.path)
        self.assertNotEqual(coverage_gaps(replace(test, grab_tolerance_x=.1)), [])

    def test_follower_walks_until_the_camera_says_it_is_there(self):
        def ball(x, y=0.):
            return np.array([approach_x(TEST) + x, y, .16])
        follower = Follower(TEST)
        self.assertEqual(follower.update(ball(.6), 0.).direction, 0)          # far: full speed
        self.assertEqual(follower.update(ball(.25), 1.).direction, FINE)      # near: fine speed
        self.assertEqual(follower.update(ball(.1), 2.).direction, FINE)
        self.assertIsNone(follower.update(ball(.07), 2.5).direction)          # within the fine coast: stop
        self.assertFalse(follower.update(ball(.02), 2.8).in_grab_zone)        # settling
        self.assertTrue(follower.update(ball(.02, .1), 3.2).in_grab_zone)     # the grab's waist twist takes y .1
        self.assertEqual(follower.update(ball(.02, .2), 3.3).direction, 2)    # sideways first, at full speed
        self.assertEqual(follower.update(ball(.02, .07), 3.6).direction, 2)   # ... until nearly centred
        self.assertEqual(follower.update(None, 3.7).direction, 2)             # out of view: toward the last sighting
        self.assertIsNone(follower.update(None, 7.8).direction)

    def test_no_correction_toward_the_balloon_smaller_than_a_step(self):
        follower = Follower(TEST)                                # min_step 0.05 > zone tolerance 0.04
        step = follower.update(np.array([approach_x(TEST) + .045, 0., .16]), 0.)
        self.assertEqual((step.direction, step.in_grab_zone), (None, True))
        self.assertEqual(Follower(TEST).update(np.array([approach_x(TEST) - .045, 0., .16]), 0.).direction, 1 + FINE)

    def test_walk_started_within_one_coast_of_the_target_still_takes_a_step(self):
        follower = Follower(TEST)
        self.assertEqual(follower.update(np.array([approach_x(TEST) + .075, 0., .16]), 0.).direction, FINE)
        self.assertEqual(follower.update(np.array([approach_x(TEST) + .045, 0., .16]), .5).direction, FINE)  # a lean
        self.assertIsNone(follower.update(np.array([approach_x(TEST) + .02, 0., .16]), 1.).direction)

    def test_fine_walk_that_does_not_start_is_restarted_then_kicked(self):
        def ball(x):
            return np.array([approach_x(TEST) + x, 0., .16])
        follower = Follower(TEST)
        self.assertEqual(follower.update(ball(-.2), 0.).direction, 1 + FINE)
        self.assertIsNone(follower.update(ball(-.19), 3.).direction)           # stalled once: stand, then restart
        self.assertEqual(follower.update(ball(-.19), 3.7).direction, 1 + FINE)
        self.assertEqual(follower.update(ball(-.19), 6.5).direction, 1)        # stalled twice: kick
        self.assertEqual(follower.update(ball(-.18), 7.).direction, 1 + FINE)
        self.assertIsNone(follower.update(ball(-.01), 7.5).direction)

    def test_safety_limits_stop_long_walks_lost_balloons_and_the_fence(self):
        far = np.array([2., 0., .16])
        follower = Follower(replace(TEST, max_walk=1.))
        self.assertEqual(follower.update(far, 0.).direction, 0)
        self.assertEqual(follower.update(far, 1.2).reason, "max_walk")
        follower = Follower(replace(TEST, lost_walk=.3))
        self.assertEqual(follower.update(far, 0.).direction, 0)
        self.assertEqual(follower.update(None, .2).direction, 0)
        self.assertEqual(follower.update(None, .4).reason, "lost")
        # 0.4 m/s dead-reckoned plus the 0.25 m forward coast: a 0.5 m fence stops the walk after ~0.5 s, the robot
        # ends inside it even after coasting, walking farther out is refused, walking back is allowed. A reset after
        # a grab keeps the dead reckoning; each planner switch adds settle_back of uncertainty.
        follower = Follower(replace(TEST, fence=.5))
        stops = [(t, follower.update(far, t)) for t in np.arange(0., 1.5, .05)]
        t, step = next((t, s) for t, s in stops if s.direction is None)
        self.assertEqual(step.reason, "fence")
        self.assertAlmostEqual(t, .5, delta=.06)
        self.assertLessEqual(follower.distance(), .5)
        follower.reset()
        self.assertEqual(follower.update(far, 3.).reason, "fence")
        self.assertEqual(follower.update(np.array([.1, 0., .16]), 5.).direction, 1 + FINE)
        follower.update(np.array([.3, 0., .16]), 5.05)   # stop the walk back (balloon now out of reach behind)
        follower.stop(5.05)
        before = follower.distance()
        for t in np.arange(5.1, 8., .1):          # a walk refused at the fence never moved: no growth
            follower.update(far, t)
        self.assertAlmostEqual(follower.distance(), before)
        follower.planner_switch()
        self.assertAlmostEqual(follower.distance(), before + TEST.settle_back)

    def test_no_full_speed_kick_with_less_than_its_coast_left(self):
        def ball(x):
            return np.array([approach_x(TEST) + x, 0., .16])
        follower = Follower(TEST)                                        # forward coast 0.25 m at full speed
        self.assertEqual(follower.update(ball(.15), 0.).direction, FINE)
        self.assertIsNone(follower.update(ball(.15), 3.).direction)      # stalled: restart
        self.assertEqual(follower.update(ball(.15), 3.7).direction, FINE)
        self.assertIsNone(follower.update(ball(.15), 6.5).direction)     # stalled again: still no kick
        self.assertEqual(follower.update(ball(.15), 7.2).direction, FINE)

    def test_never_walks_toward_a_balloon_closer_than_the_limit(self):
        follower = Follower(TEST)
        step = follower.update(np.array([.28, 0., .16]), 0., target_x=.1, tolerance_x=.05)
        self.assertEqual((step.direction, step.reason), (None, "too_close"))

    def test_no_blind_start_near_the_target(self):
        follower = Follower(replace(TEST, lost_walk=4.))
        follower.update(np.array([approach_x(TEST) + .2, 0., .16]), 0.)       # seen, starts a fine walk
        follower.stop(.1)
        self.assertIsNone(follower.update(None, 2.).direction)            # lost: no new walk from the old sighting

    def test_robot_follow_needs_declared_commands_and_a_measured_robot_calibration(self):
        robot = calibration.load(calibration.for_backend("robot"))
        self.assertEqual(robot.unmeasured(), ["camera", "balloon", "grab", "walk"])
        self.assertEqual(SIM.unmeasured(), [])
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            problems = follow_problems("robot", build, calibration.for_backend("robot"))
            self.assertEqual(len(problems), 4, problems)   # no commands, no speeds (x2), four unmeasured sections
            (build / "CAPABILITIES").write_text("reference\nfine-move\nwalk-speed 0.4\nfine-walk-speed 0.15\n")
            problems = follow_problems("sim", build, calibration.for_backend("sim"))
            self.assertEqual(len(problems), 2, problems)   # planner-stance missing; fine speed differs
            (build / "CAPABILITIES").write_text("reference\nfine-move\nplanner-stance\nwalk-speed 0.4\n"
                                                "fine-walk-speed 0.2\n")
            self.assertEqual(follow_problems("sim", build, calibration.for_backend("sim")), [])
            self.assertIn("not the robot's", " ".join(follow_problems("robot", build, calibration.for_backend("sim"))))
            data = json.loads(calibration.for_backend("robot").read_text())
            for section in calibration.MEASURED_SECTIONS:
                data[section]["measured"] = "2026-11-04 test"
            (build / "robot.json").write_text(json.dumps(data))
            self.assertEqual(follow_problems("robot", build, build / "robot.json"), [])


if __name__ == "__main__":
    unittest.main()
