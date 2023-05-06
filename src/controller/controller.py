class PIDController:

    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max

        self.last_error = 0
        self.integral = 0

    def compute(self, error, dt):
        derivative = (error - self.last_error) / dt
        self.integral += error * dt
        self.last_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = max(self.out_min, min(self.out_max, output))  # clamp output
        
        return output