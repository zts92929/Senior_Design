from huesdk import Hue
import time


username = Hue.connect(bridge_ip = "10.0.0.113")
print(username)

hue = Hue("10.0.0.113", username)

light = hue.get_light(name="4873_Bulb_1")

brightness_values = [10, 25, 50, 100, 150, 200]
for i in brightness_values:
	light.set_brightness(i)
	print(i)
	time.sleep(3)

light.set_brightness(10)