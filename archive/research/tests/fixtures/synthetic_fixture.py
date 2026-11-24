"""
Hand-computed test fixtures for unit tests and smoke tests.
"""

from typing import List, Dict, Any
from src.data.schema import TurnRecord, QuestionRecord


def get_tiny_synthetic_fixture():
    """
    Hand-crafted 2 conversations with known evidence turns for exact metric verification.
    Conv 1: 10 turns, 10 tokens each.
      Turn 2: Evidence for Q1 (category: single_fact)
      Turn 6: Evidence for Q1 and Q2 (category: multi_hop)
    Conv 2: 8 turns, 10 tokens each.
      Turn 3: Evidence for Q3 (category: temporal)
    """
    turns = [
        # Conv 1
        TurnRecord("c1", "s1", 0, 0, "user", "Hello assistant", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s1", 1, 1, "assistant", "Hello user", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s1", 2, 2, "user", "My secret code is 4421.", 10, 1, ["q1"], "single_fact", "longmemeval", "en"),
        TurnRecord("c1", "s1", 3, 3, "assistant", "I noted your code.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s2", 0, 4, "user", "What is the weather?", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s2", 1, 5, "assistant", "It is sunny.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s2", 2, 6, "user", "My backup pin is 9988.", 10, 1, ["q1", "q2"], "multi_hop", "longmemeval", "en"),
        TurnRecord("c1", "s2", 3, 7, "assistant", "Backup pin recorded.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s2", 4, 8, "user", "Thanks for helping.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c1", "s2", 5, 9, "assistant", "Anytime!", 10, 0, [], "none", "longmemeval", "en"),
        # Conv 2
        TurnRecord("c2", "s1", 0, 0, "user", "Starting new session", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 1, 1, "assistant", "Understood", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 2, 2, "user", "How are things?", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 3, 3, "assistant", "The meeting was moved to Friday.", 10, 1, ["q3"], "temporal", "longmemeval", "en"),
        TurnRecord("c2", "s1", 4, 4, "user", "Okay good to know.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 5, 5, "assistant", "See you then.", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 6, 6, "user", "Bye", 10, 0, [], "none", "longmemeval", "en"),
        TurnRecord("c2", "s1", 7, 7, "assistant", "Goodbye", 10, 0, [], "none", "longmemeval", "en"),
    ]

    questions = [
        QuestionRecord("q1", "c1", "single_fact", ["c1_s1_2", "c1_s2_2"], "What are user codes?"),
        QuestionRecord("q2", "c1", "multi_hop", ["c1_s2_2"], "What is the backup pin?"),
        QuestionRecord("q3", "c2", "temporal", ["c2_s1_3"], "When is the meeting?"),
    ]

    return turns, questions

