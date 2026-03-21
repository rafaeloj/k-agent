from abc import ABC, abstractmethod
import typing as T

if T.TYPE_CHECKING:
    from selection_agent.agent.schemas import ClientSchema

class Selection(ABC):
    
    @abstractmethod
    def sample_clients(self, clients: T.List['ClientSchema'], sample_size: int, **kargs) -> T.Dict[str, T.List['ClientSchema']|str]:
        """
        Sample a subset of clients from the provided list.
        Args:
            clients: A list of ClientSchema objects to sample from.
            sample_size: The number of clients to include in the sample.

        Returns:
            A dictionary containing:
                - A list of sampled ClientSchema objects
                - A status message string indicating the result of the sampling operation

        Raises:
            NotImplementedError: This is an abstract method and must be implemented by subclasses.
        """