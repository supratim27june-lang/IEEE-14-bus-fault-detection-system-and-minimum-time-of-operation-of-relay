import numpy as np

class Relay:

    def __init__(self):
        pass

    def relay_operating_time(self, fault_current, pickup_current, tds):

        multiple = fault_current / pickup_current

        # Relay should not trip
        if multiple <= 1:
            return 9999

        operating_time = (
            0.14 * tds
        ) / (np.power(multiple, 0.02) - 1)

        return operating_time