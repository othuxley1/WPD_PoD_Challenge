"""
A script to write a battery scheduler for the PoD Data Science Challenge

- First Authored: 2021-01-30
- Owen Huxley <othuxley1@sheffield.ac.uk>
"""

import pandas as pd

class BatteryScheduler:

    def __init__(self, max_import, max_export, capacity, charge_coeff,
                 initial_demand, PV_gen, initial_SOC):
        self.__max_import = max_import  # +ve
        self.__max_export = max_export  # -ve
        self.__capacity = capacity
        self.__charge_coeff = charge_coeff
        self.demand = initial_demand
        self.__PV_gen = PV_gen
        self.current_SOC = initial_SOC # state of charge

    def step(self):
        return

    def charge(self):
        return

    def discharge(self):
        return





