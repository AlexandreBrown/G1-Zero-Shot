"""catch_balloon's held balloon, placed beyond arm's reach so the robot has to walk to it first."""
from gpt_astra.tasks.catch_balloon import sim as balloon

# Pelvis frame at spawn (x forward, y = robot's left); the holder then keeps it still in the world at the
# reach clips' palm height. Reaches touch around x 0.45, so every point needs 0.2-0.6 m of walking.
FAR_POINTS = ((.85, 0.), (.75, .17), (.95, -.17), (.70, -.30), (1.05, .10),
              (.80, .30), (.65, -.05), (1.00, -.25), (.90, .20))
balloon.HELD = balloon.HOVER = True
balloon.HOVER_POINTS = tuple((x, y, balloon.HELD_Z) for x, y in FAR_POINTS)
balloon.HOVER_SECONDS = 90.0
balloon.HOVER_STIFFNESS, balloon.HOVER_DAMPING = 40.0, 2.5
prepare_scene, TaskSimulation = balloon.prepare_scene, balloon.TaskSimulation
