from .agents_base import AgentBase

class SummarizeTool(AgentBase):
    def __init__(self, max_retries, verbose=True):
        super.__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)
        