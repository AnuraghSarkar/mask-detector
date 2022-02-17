from gpiozero import Buzzer, LED
from time import sleep
import board
import adafruit_mlx90614
buzzer = Buzzer(21)
red = LED(14)
green = LED(15)
i2c = board.I2C()
mlx = adafruit_mlx90614.MLX90614(i2c)

while True:
    buzzer.on()
    green.off()
    red.on()
    print("Object Temp: ", (mlx.object_temperature*9/5)+32)

    sleep(1)
    buzzer.off()
    red.off()
    green.on()
    print("Object Temp: ", (mlx.object_temperature*9/5)+32)

    sleep(1)
