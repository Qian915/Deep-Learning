from pattern import Checker
from pattern import Circle
from generator import ImageGenerator

# Checkerboard
a = Checker(250, 25)
a.draw()
a.show()

# Circle
b = Circle(1024, 200, (512, 256))
b.draw()
b.show()

# Image generator
ima_gen = ImageGenerator('./exercise_data/', './Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
ima_gen.show(resize=True)
