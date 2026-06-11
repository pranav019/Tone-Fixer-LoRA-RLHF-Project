# ── Imports ──────────────────────────────────────────────────
import json  # for saving data as JSON file
import os  # for creating folders
from datasets import Dataset  # HuggingFace Dataset class
import pandas as pd  # for displaying data as table

# ── Step 1: Define raw data ───────────────────────────────────
# Each entry has:
#   "rude"   → the input the model receives
#   "polite" → the target output the model must learn to produce

raw_data = [
    # ── Work / Office Context ─────────────────────────────────
    {
        "rude": "This report is trash. Do it again.",
        "polite": "This report needs significant improvement. Could you please revise it?",
    },
    {
        "rude": "Why haven't you finished this yet? You're so slow.",
        "polite": "I noticed this task is still pending. Is there anything blocking your progress?",
    },
    {
        "rude": "Your presentation was boring and made no sense.",
        "polite": "The presentation could benefit from clearer structure and more engaging content.",
    },
    {
        "rude": "Stop wasting everyone's time with stupid questions.",
        "polite": "Let's try to keep questions focused so we can make the most of everyone's time.",
    },
    {
        "rude": "You clearly have no idea what you're doing.",
        "polite": "It seems like you might need some additional guidance on this task.",
    },
    {
        "rude": "I don't care about your excuses. Just get it done.",
        "polite": "I understand there may be challenges, but it's important we meet this deadline.",
    },
    {
        "rude": "This is the worst code I have ever seen in my life.",
        "polite": "This code needs significant refactoring. Let's discuss some improvements.",
    },
    {
        "rude": "You missed the deadline again. Unbelievable.",
        "polite": "I noticed the deadline was missed again. Can we discuss how to prevent this?",
    },
    {
        "rude": "Nobody wants to hear your opinion right now.",
        "polite": "Let's hear from others first, and then we can come back to your thoughts.",
    },
    {
        "rude": "Fix this bug immediately or you're off the project.",
        "polite": "This bug is a priority. Please address it as soon as possible.",
    },
    # ── Email / Communication Context ─────────────────────────
    {
        "rude": "Why haven't you replied to my email? It's been days.",
        "polite": "I wanted to follow up on my previous email. Could you please respond when you get a chance?",
    },
    {
        "rude": "Your email made absolutely no sense whatsoever.",
        "polite": "I had some difficulty understanding your email. Could you please clarify?",
    },
    {
        "rude": "Stop sending me useless updates. I don't need them.",
        "polite": "Could we streamline our updates to focus only on key developments?",
    },
    {
        "rude": "This is not what I asked for at all. Read properly.",
        "polite": "This doesn't quite match what was requested. Could you please review the requirements?",
    },
    {
        "rude": "You always send incomplete information. So annoying.",
        "polite": "It would be helpful to receive complete information in future communications.",
    },
    # ── Team / Collaboration Context ──────────────────────────
    {
        "rude": "Your idea is stupid and won't work.",
        "polite": "I have some concerns about this approach. Could we explore alternative solutions?",
    },
    {
        "rude": "You're dragging the whole team down.",
        "polite": "I think we need to discuss how we can better align your work with the team's goals.",
    },
    {
        "rude": "Stop interrupting me when I'm talking.",
        "polite": "I'd appreciate the opportunity to finish my thought before we discuss further.",
    },
    {
        "rude": "You never listen to anything anyone says.",
        "polite": "I feel like some of my points may not be coming across clearly. Can we revisit them?",
    },
    {
        "rude": "This team is completely incompetent.",
        "polite": "I think we have some opportunities to improve our team's performance and processes.",
    },
    # ── Customer / Client Context ─────────────────────────────
    {
        "rude": "Your product is garbage and a waste of money.",
        "polite": "I'm disappointed with the product and feel it didn't meet my expectations.",
    },
    {
        "rude": "Your customer service is absolutely terrible.",
        "polite": "I've had a frustrating experience with customer support and would appreciate better assistance.",
    },
    {
        "rude": "I've been waiting forever. This is ridiculous.",
        "polite": "I've been waiting for quite some time. Could you provide an update on my request?",
    },
    {
        "rude": "Nobody here knows what they're doing.",
        "polite": "I feel the support I received wasn't adequate. Could I speak with someone more experienced?",
    },
    {
        "rude": "This is the last time I use your useless service.",
        "polite": "I've been quite dissatisfied with the service and am considering other options.",
    },
    # ── Feedback / Review Context ─────────────────────────────
    {
        "rude": "This design looks absolutely horrible.",
        "polite": "I think this design could be improved. Could we explore some alternative directions?",
    },
    {
        "rude": "Did you even try on this assignment?",
        "polite": "This assignment doesn't quite meet the expected standard. Let's discuss how to improve it.",
    },
    {
        "rude": "Your writing is terrible and hard to read.",
        "polite": "The writing could benefit from improved clarity and structure.",
    },
    {
        "rude": "This is a complete waste of my time to review.",
        "polite": "This work needs more preparation before it's ready for review.",
    },
    {
        "rude": "How did you even get hired with skills like this?",
        "polite": "I think there are some skill areas we should focus on developing together.",
    },
]

# ── Step 2: Format data for training ─────────────────────────
# We format each pair as a single string that GPT-2 will learn
# The model sees the full text and learns to complete it
# Format: "Rude: [rude text] Polite: [polite text]"

formatted_data = []  # empty list to hold formatted examples

for item in raw_data:
    # Create a single training string from each pair
    # The model learns: given "Rude: X Polite:", complete with Y
    text = f"Rude: {item['rude']} Polite: {item['polite']}"

    formatted_data.append(
        {
            "rude": item["rude"],  # original rude sentence
            "polite": item["polite"],  # target polite sentence
            "text": text,  # combined training string
        }
    )

# ── Step 3: Split into train and test sets ────────────────────
# We use 80% for training, 20% for testing
# This lets us evaluate how well the model generalizes

split_index = int(len(formatted_data) * 0.8)  # 80% cutoff point

train_data = formatted_data[:split_index]  # first 80%
test_data = formatted_data[split_index:]  # last 20%

print(f"Total examples : {len(formatted_data)}")
print(f"Training set   : {len(train_data)} examples")
print(f"Test set       : {len(test_data)} examples")

# ── Step 4: Convert to HuggingFace Dataset ────────────────────
# HuggingFace's Dataset class is what the Trainer expects
# It's like a pandas DataFrame but optimized for ML training

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# ── Step 5: Save to disk ──────────────────────────────────────
# Save as JSON files in the data/ folder
# These will be uploaded to Colab for training

os.makedirs("data", exist_ok=True)  # create folder if not exists

# Save raw JSON (human readable, easy to inspect)
with open("data/train.json", "w") as f:
    json.dump(train_data, f, indent=2)  # indent=2 makes it pretty

with open("data/test.json", "w") as f:
    json.dump(test_data, f, indent=2)

# Save as HuggingFace Dataset format (faster loading during training)
train_dataset.save_to_disk("data/train_dataset")
test_dataset.save_to_disk("data/test_dataset")

print("\n✅ Dataset saved successfully!")
print("   data/train.json       ← human readable")
print("   data/test.json        ← human readable")
print("   data/train_dataset/   ← HuggingFace format")
print("   data/test_dataset/    ← HuggingFace format")

# ── Step 6: Preview the dataset ───────────────────────────────
# Display a few examples so we can verify it looks correct

print("\n" + "=" * 60)
print("DATASET PREVIEW — First 3 Training Examples")
print("=" * 60)

df = pd.DataFrame(train_data)  # convert to DataFrame for display

for i, row in df.head(3).iterrows():
    print(f"\nExample {i+1}:")
    print(f"  RUDE   : {row['rude']}")
    print(f"  POLITE : {row['polite']}")
    print(f"  TEXT   : {row['text']}")
    print("-" * 60)

# ── Step 7: Dataset statistics ────────────────────────────────
print("\n📊 Dataset Statistics:")
print(
    f"  Average rude sentence length   : "
    f"{df['rude'].str.len().mean():.1f} characters"
)
print(
    f"  Average polite sentence length : "
    f"{df['polite'].str.len().mean():.1f} characters"
)
print(f"  Shortest example               : " f"{df['text'].str.len().min()} characters")
print(f"  Longest example                : " f"{df['text'].str.len().max()} characters")
