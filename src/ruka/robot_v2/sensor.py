from abc import ABC, abstractmethod
from typing import Any, Dict


class SensorSystem(ABC):
    @abstractmethod
    def capture(self) -> Dict[str, Any]:
        '''
        Immidiately returns the last set of frames captured.

        Returns:
            frames: dict, mapping sensor id to frame
                sensor id is composed like {sensor_type}_{sensor_serial}
        '''

        pass
