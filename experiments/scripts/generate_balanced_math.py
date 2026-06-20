import json
import random
import os

data = []
# Generate 50 correct
for i in range(50):
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    data.append({
        "id": f"math_balanced_{i}",
        "problem": f"What is {a} + {b}?",
        "solution": f"The sum of {a} and {b} is {a+b}. #### {a+b}",
        "label": 1
    })

# Generate 50 incorrect
for i in range(50):
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    wrong_ans = a + b + random.randint(1, 10)
    data.append({
        "id": f"math_balanced_{i+50}",
        "problem": f"What is {a} + {b}?",
        "solution": f"The sum of {a} and {b} is {wrong_ans}. #### {wrong_ans}",
        "label": 0
    })

os.makedirs("data/raw/synthetic_math", exist_ok=True)
with open("data/raw/synthetic_math/balanced_100.json", "w") as f:
    json.dump(data, f, indent=2)

print("Generated 100 balanced examples in data/raw/synthetic_math/balanced_100.json")
