import unittest
from agentic_ai.core import Agent
from agentic_ai.memory import Memory

class TestAgent(unittest.TestCase):
    def test_think(self):
        agent = Agent(name="AI", memory=Memory())
        response = agent.think("Hello")
        self.assertIn("thinking about", response)

if __name__ == "__main__":
    unittest.main()
