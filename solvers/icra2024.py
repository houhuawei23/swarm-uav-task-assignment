from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
import numpy as np


from framework import *


class ICRA2024_CoalitionFormationGame(CoalitionFormationGame):
    """
    ```cpp
    // Alg1: Decision-Making Alg for each ri

    Partition Alg1(R /*a set of I robots*/, T /*a set of J tasks*/, D /*a set of neighbors of the robots */) {
    // Initialize:
    // r_satisfied ← 0;
    // ξi ← 0;
    // Πi ← {Sϕ = R, Sj = ∅ ∀tj ∈ T};
    while(r_satisfied == true) {
        if(r_satisfied_i == 0) {
        // Based on Equation (2), calculate the utility to
        // determine the Sj∗ and tj∗ that maximize its
        // utility;
        if ui(tj*, Sj*) > ui(txx, Sxx) {
            // joinSj*, update Πi
            ξi = ξi + 1;
        }
        r_satisfied_i = 1;
        }
        // Broadcast Mi = {Πi, r_satisfied, ξi} and
        // receive Mk from its neighbors Di
        // Collect all the messages and construct
        // Mi_all = {Mi, Mk}
        for(each message Mk in Mi_all) {
        if (ξk >= ξi) {
            Mi = Mk;
            if (Πi != Πk) r_satisfied_i = 0;
        }
        }
    }
    if (r_satisfied == 1) return Π0; //?
    }
    ```
    """

    def run_allocate(self, debug=False):
        pass
