import os
import platform

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if platform.system() == "Windows":
    os.environ['path'] += (";" + os.path.join(current_dir, 'lib/models/pose'))

from lib.pose.poselandmark import PoseLandmark
