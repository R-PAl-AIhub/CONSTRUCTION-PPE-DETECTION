from collections import deque
import time

class RiskEngine:
    def __init__(self):
        self.violation = deque()
    
    def add_violation(self,violation_type:str):
        self.violation.append(
            {
                "type":violation_type, 
                "time":time.time()
            }
        )