"""Turn agent trajectories into (text, label) examples, split by trajectory, and chunk."""
from sklearn.model_selection import train_test_split

# Trajectory fields to concatenate; adjust to match the AgentDoG schema.
FIELDS = ["prompt", "tool_calls", "actions", "final_answer"]


def trajectory_to_text(traj):
    parts = []
    for field in FIELDS:
        value = traj.get(field)
        if value:
            parts.append(f"[{field.upper()}] {value}")
    return "\n".join(parts)


def split(examples, seed=0):
    """80/10/10 stratified split at the trajectory level (before chunking)."""
    labels = [y for _, y in examples]
    train, rest = train_test_split(examples, test_size=0.2, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=[y for _, y in rest], random_state=seed)
    return train, val, test


def chunk(text, tokenizer, max_len=512):
    """Split one trajectory into token chunks of at most max_len (incl. special tokens)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    step = max_len - 2
    return [tokenizer.build_inputs_with_special_tokens(ids[i:i + step])
            for i in range(0, max(len(ids), 1), step)]
