from scipy.constants import speed_of_light
import numpy as np

def getshift(freqs,isotope,measured_voltage):
    vals_shifted_down = []
    vals_shifted_up = []
    values = freqs # THz
    voltage = np.abs(measured_voltage) # V
    mass = isotope # in amu
    amu_to_kg = 1.6605418E-27
    fundamental_charge = 1.60218E-19

    beta = np.sqrt((2*voltage*fundamental_charge)/((mass)*amu_to_kg)) / speed_of_light #180000/speed_of_light
    # for value in values: 
    shifted_value_down = freqs * (1 - beta) / np.sqrt(1 - beta**2)
    # shifted_value_up = freqs * (1 + beta) / np.sqrt(1 + beta**2)

        # vals_shifted_down.append(shifted_value_down)
        # vals_shifted_up.append(shifted_value_up)

    return shifted_value_down