
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.last_error = 0
        self.integral = 0

    def compute(self, error, dt):
        derivative = error - self.last_error / dt
        self.integral += error * dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_error = error

        return output
