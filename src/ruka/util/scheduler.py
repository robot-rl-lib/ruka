from typing import Dict

class Scheduler:
    """ Simpple scheduler for values """
    
    def __init__(self, table: Dict[int, float]):
        """
        params:
            table: {step: value} like  {0, 0.1 , 100: 0.01, ...}
        """
        self._table = table.copy()
        self._steps = list(sorted(self._table.keys()))
        assert len(self._steps) > 0
        self._cur = 0
        
        
    def value(self, step):
        # TODO: support step decrease
        if self._cur + 1 < len(self._steps) and step >= self._steps[self._cur + 1]:
            self._cur += 1
        return self._table[self._steps[self._cur]]
