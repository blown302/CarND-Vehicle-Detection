class KalmanFilter:
    def __init__(self):
        self.varience = 1000
        self.mean = 0
        self.measurement_varience = 4.
        self.motion_varience = 2.

    def update_probability(self, measurement):
        new_mean = float(self.measurement_varience * self.mean + self.varience * measurement) / (self.varience + self.measurement_varience)
        new_var = 1./(1./self.varience + 1./self.measurement_varience)
        return (new_mean, new_var)

    def update(self, measurement, motion):
        mean2, var2 = self.update_probability(measurement)
        self.mean = motion + mean2
        self.varience = self.motion_varience + var2
    
    def get_value(self):
        return self.mean
    
    def get_varience(self):
        return self.varience