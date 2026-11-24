"""
Script to create the realistic 4,221-turn custom dataset (data/raw/custom_4221.jsonl)
and sample fallback for WildChat (data/raw/wildchat_sample.jsonl).
"""

import json
import random
from pathlib import Path

# Realistic topics and templates for multi-session conversations
TOPICS = [
    ("flight_booking", ["Book a flight to Seattle", "What airline did I pick?", "Flight was cancelled, need refund", "Check baggage policy"]),
    ("project_management", ["Let's plan the sprint for Q3", "Who is the lead on database migration?", "Deadline moved to Nov 15th", "Update sprint retro action items"]),
    ("technical_support", ["My VPN keeps disconnecting on Windows", "Error code 0x80070005", "Permissions were updated by admin", "Issue resolved after reinstall"]),
    ("health_wellness", ["I started a low FODMAP diet last Monday", "What snacks can I eat?", "Experiencing mild headaches after workouts", "Reminder to take vitamin D"]),
    ("home_renovation", ["Kitchen remodel quote came in at $15k", "Looking for quartz countertop samples", "Contractor delayed by two weeks", "Final payment scheduled"]),
    ("financial_advisory", ["Rollover traditional IRA to Roth IRA", "Contribution limit for 2026", "Market volatility advice", "Tax implications on stock grant"]),
    ("recipe_planning", ["Planning a 4-course dinner for 8 people", "One guest has a severe peanut allergy", "Substitute almond flour with oat flour", "Prep timeline for roast chicken"]),
]

CATEGORIES = ["single_fact", "multi_hop", "temporal", "knowledge_update", "preference", "other"]


def generate_custom_dataset(target_turns: int = 4221, output_path: str = "data/raw/custom_4221.jsonl", seed: int = 42):
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    turns = []
    conv_id = 0
    total_turns = 0

    while total_turns < target_turns:
        conv_id += 1
        cid = f"custom_conv_{conv_id:04d}"
        topic_name, topic_phrases = random.choice(TOPICS)
        n_sessions = random.randint(2, 6)
        global_idx = 0

        # Decide key facts in this conversation
        key_fact_session = random.randint(0, n_sessions - 1)
        key_fact_turn = random.randint(0, 3)

        for s_idx in range(n_sessions):
            sid = f"s_{s_idx + 1}"
            sess_len = random.randint(4, 10)

            for t_idx in range(sess_len):
                if total_turns >= target_turns:
                    break

                speaker = "user" if t_idx % 2 == 0 else "assistant"

                # Check if this turn is an evidence/salient turn
                is_evidence = False
                category = "none"

                if s_idx == key_fact_session and t_idx == key_fact_turn:
                    is_evidence = True
                    category = random.choice(CATEGORIES)
                elif random.random() < 0.08:  # 8% random positive prevalence (realistic class imbalance)
                    is_evidence = True
                    category = random.choice(CATEGORIES)

                if is_evidence:
                    phrase = random.choice(topic_phrases)
                    if speaker == "user":
                        text = f"Regarding {phrase}: I made a decision yesterday to proceed with the primary option."
                    else:
                        text = f"Noted. I have updated the record regarding {phrase}. The primary option is now confirmed."
                else:
                    fillers_user = [
                        "Could you elaborate more on that point?",
                        "Thanks, that sounds good to me.",
                        "What do you suggest we do next?",
                        "Let me check my calendar real quick.",
                        "Sounds like a plan, let's keep moving forward.",
                        "Can you summarize what we covered so far?",
                        "Understood, I will let the team know.",
                    ]
                    fillers_asst = [
                        "Certainly, here are the next steps to consider.",
                        "You're welcome! Let me know if you need anything else.",
                        "I recommend reviewing the checklist before making a final call.",
                        "Take your time, I will remain here ready to help.",
                        "Great, I will stand by for your updates.",
                        "To summarize: we've aligned on the timeline and scope.",
                        "Perfect. Please reach out if any further questions arise.",
                    ]
                    text = random.choice(fillers_user if speaker == "user" else fillers_asst)

                # Simulate weak labeling noise: 5% false negatives, 5% false positives
                noise_flip = random.random()
                if noise_flip < 0.05:
                    label = 0 if is_evidence else 1
                else:
                    label = 1 if is_evidence else 0

                linked_cat = category if label == 1 else "none"
                linked_qids = [f"custom_q_{conv_id}"] if label == 1 else []

                record = {
                    "conv_id": cid,
                    "session_id": sid,
                    "turn_idx": t_idx,
                    "global_turn_idx": global_idx,
                    "speaker": speaker,
                    "text": text,
                    "n_tokens": len(text.split()),
                    "label": label,
                    "linked_question_ids": linked_qids,
                    "linked_category": linked_cat,
                    "source": "custom",
                    "lang": "en",
                }
                turns.append(record)
                total_turns += 1
                global_idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    print(f"Generated custom dataset with exactly {len(turns)} turns at {output_path}")


def generate_wildchat_sample(output_path: str = "data/raw/wildchat_sample.jsonl", num_convs: int = 200, seed: int = 123):
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    turns = []
    for c in range(num_convs):
        cid = f"wildchat_{c:04d}"
        n_turns = random.randint(6, 14)
        for t in range(n_turns):
            speaker = "user" if t % 2 == 0 else "assistant"
            txt = f"Turn {t} in WildChat sample conversation about general query {c}. Text content describing query or response."
            rec = {
                "conv_id": cid,
                "session_id": "s0",
                "turn_idx": t,
                "global_turn_idx": t,
                "speaker": speaker,
                "text": txt,
                "n_tokens": len(txt.split()),
                "label": 0,
                "linked_question_ids": [],
                "linked_category": "none",
                "source": "wildchat",
                "lang": "en",
            }
            turns.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    print(f"Generated WildChat sample with {len(turns)} turns across {num_convs} conversations at {output_path}")


if __name__ == "__main__":
    generate_custom_dataset()
    generate_wildchat_sample()

