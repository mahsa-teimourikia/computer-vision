# Knowledge checks

The [Learning Hub](../hub/index.html#curriculum) contains one focused checkpoint for every published course:

- 9 Beginner checkpoints;
- 6 Intermediate checkpoints; and
- 9 Advanced checkpoints.

Together, the 24 checkpoints currently contain 137 selectable-answer questions. They test mechanisms, evaluation semantics, failure diagnosis, architecture choices, operational boundaries, and production judgment—not only vocabulary.

## How scoring works

1. Open a course in the Hub and select **Checkpoint**.
2. Answer every question before submitting.
3. The Hub reports the score and shows the defensible answer for every item as concise review feedback.
4. A course is marked complete only after a perfect checkpoint score.
5. Completion progress is stored in that browser with `localStorage`; it is not sent to a server and is not evidence of certification.

Use **Reset progress** in the curriculum header to clear the local completion record. Clearing browser storage has the same effect.

## Validation contract

`make quiz-check` verifies that:

- every available course has exactly one Hub workspace and checkpoint;
- every checkpoint has at least five questions;
- every question has at least three options and exactly one defensible answer;
- radio names are unique within a checkpoint;
- answer-level feedback and saved-progress behavior are present; and
- planned Enterprise and Capstone cards are not presented as completed lessons.

The checkpoints complement the notebook exercises. They do not replace execution evidence, instructor review, or capability evaluation.
