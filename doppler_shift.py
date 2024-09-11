from scipy.constants import speed_of_light
import numpy as np

def getshift(wls,isotope):
    vals_shifted_down = []
    values = wls # nm
    voltage = 30000 # eV
    mass = isotope # in amu
    amu_to_kg = 1.6605418E-27
    fundamental_charge = 1.60218E-19

    beta = np.sqrt((2*voltage*fundamental_charge)/((mass)*amu_to_kg)) / speed_of_light #180000/speed_of_light
    for value in values: 
        shifted_value_down = value * (1 - beta) / np.sqrt(1 - beta**2)
        shifted_value_up = value * (1 + beta) / np.sqrt(1 + beta**2)

        vals_shifted_down.append(shifted_value_down)


    # print('beta =', beta)
    # print('original value =', value)
    # print(mass)
    # print('shifted value down =', shifted_value_down)
    # print('shifted value up =', shifted_value_up)

    return vals_shifted_down