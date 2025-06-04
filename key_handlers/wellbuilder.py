# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:10:47 2025

@author: patri
"""

class WellBuilder:
    def __init__(self):
        self._wells = None  # Store the wells, initially not set
        self.rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
       
    
    def _build_96_wells(self):
        total_wells = []
        for x in self.rows:
            for y in self.columns:
                total_wells += [x + y]
        return total_wells

    def get_wells(self):
        if self._wells is None:  # Only build the wells if they aren't initialized yet
            self._wells = self._build_96_wells()
        return self._wells