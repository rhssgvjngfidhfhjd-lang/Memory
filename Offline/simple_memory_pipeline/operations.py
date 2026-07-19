from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Operation:
    name: str
    description: str
    instruction_template: str
    update_type: str


_DEFAULT_OPERATIONS = [
    Operation(
        name="insert",
        description=(
            "Insert a new memory when the current chunk contains information "
            "that is not already stored and may be needed to answer questions later."
        ),
        instruction_template="""Operation: Insert New Memory
Purpose: Capture new information from the current chunk that is missing from memory.
When to use:
- The chunk introduces new facts about the user: events, preferences, plans, or stable context.
- The chunk contains salient conversation content: topics discussed, questions asked, explanations or opinions given, and conclusions reached.
How to apply:
- Compare against retrieved memories to avoid duplicates.
- Split unrelated facts into separate memory items.
- Preserve specifics needed to answer questions later: key entities, numbers, dates, and references to images.
- Keep each memory item concise but complete.
Constraints:
- Do not update or delete an existing memory item.
- Skip greetings, small talk, and filler.
Action type: INSERT only.
""",
        update_type="insert",
    ),
    Operation(
        name="update",
        description=(
            "Update an existing memory when the current chunk corrects, extends, or "
            "clarifies a retrieved memory."
        ),
        instruction_template="""Operation: Update Existing Memory
Purpose: Revise a retrieved memory with corrected or newer information from the current chunk.
When to use:
- The chunk clarifies, corrects, or extends a retrieved memory.
How to apply:
- Select the best matching retrieved memory item.
- Merge old and new details into one updated memory item.
- Preserve details that still remain true.
Constraints:
- Do not create a new memory item.
- Do not delete an item.
Action type: UPDATE only.
""",
        update_type="update",
    ),
    Operation(
        name="delete",
        description=(
            "Delete a memory when the current chunk clearly shows that a retrieved "
            "memory is wrong, obsolete, canceled, or superseded."
        ),
        instruction_template="""Operation: Delete Invalid Memory
Purpose: Remove a retrieved memory that is clearly wrong, outdated, canceled, or superseded.
When to use:
- The chunk explicitly contradicts a memory.
- A plan, fact, or status is explicitly canceled or replaced.
How to apply:
- Delete only when the evidence is explicit.
Constraints:
- If uncertain, prefer NOOP over DELETE.
- Do not insert or update in this operation block.
Action type: DELETE only.
""",
        update_type="delete",
    ),
    Operation(
        name="noop",
        description="Return NOOP when the current chunk does not require any memory change.",
        instruction_template="""Operation: No Operation
Purpose: Confirm that no memory change is needed for the current chunk.
When to use:
- The chunk contains only greetings, small talk, or filler.
- Everything informative in the chunk is already covered by existing memories.
Constraints:
- Use NOOP only if INSERT, UPDATE, and DELETE are all unnecessary.
- If uncertain whether the chunk's content is worth storing, prefer INSERT over NOOP.
Action type: NOOP only.
""",
        update_type="noop",
    ),
]


def get_default_operations() -> List[Operation]:
    return list(_DEFAULT_OPERATIONS)


def get_operations(names: Sequence[str]) -> List[Operation]:
    selected = {str(name).strip().lower() for name in names if str(name).strip()}
    known = {operation.name for operation in _DEFAULT_OPERATIONS}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown operations: {sorted(unknown)}. Available: {sorted(known)}")
    operations = [operation for operation in _DEFAULT_OPERATIONS if operation.name in selected]
    if not operations:
        raise ValueError("At least one operation must be enabled.")
    return operations
