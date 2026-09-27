const STORAGE_KEY = "computer-vision-field-guide-progress-v1";
const levelButtons = document.querySelectorAll("[data-level][role='tab']");
const cards = document.querySelectorAll(".lesson-card");
const availableCards = document.querySelectorAll("button.lesson-card[data-workspace]");
const workspaces = document.querySelectorAll("[data-workspace-panel]");
const progressCount = document.querySelector("#progress-count");
const progressBar = document.querySelector("#course-progress");
const resetProgress = document.querySelector("#reset-progress");

function loadProgress() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {
      completed: new Set(Array.isArray(parsed.completed) ? parsed.completed : []),
      selectedCourse: typeof parsed.selectedCourse === "string" ? parsed.selectedCourse : "foundations",
      selectedLevel: typeof parsed.selectedLevel === "string" ? parsed.selectedLevel : "all",
    };
  } catch {
    return { completed: new Set(), selectedCourse: "foundations", selectedLevel: "all" };
  }
}

const progressState = loadProgress();

function saveProgress() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      completed: [...progressState.completed].sort(),
      selectedCourse: progressState.selectedCourse,
      selectedLevel: progressState.selectedLevel,
    }));
  } catch {
    // The Hub remains fully usable when storage is disabled.
  }
}

function updateProgress() {
  const count = [...progressState.completed].filter((id) =>
    document.querySelector(`button.lesson-card[data-workspace="${id}"]`)
  ).length;
  progressCount.textContent = `${count} / ${availableCards.length}`;
  progressBar.max = availableCards.length;
  progressBar.value = count;
  progressBar.textContent = `${count} of ${availableCards.length}`;
  for (const card of availableCards) {
    const complete = progressState.completed.has(card.dataset.workspace);
    card.classList.toggle("completed", complete);
    const status = card.querySelector(".status");
    if (status) status.textContent = complete ? "COMPLETE" : "AVAILABLE";
  }
}

function selectLevel(selected) {
  const valid = [...levelButtons].some((button) => button.dataset.level === selected);
  const level = valid ? selected : "all";
  for (const button of levelButtons) button.setAttribute("aria-selected", String(button.dataset.level === level));
  for (const card of cards) card.hidden = level !== "all" && card.dataset.level !== level;
  progressState.selectedLevel = level;
  saveProgress();
}

for (const button of levelButtons) {
  button.addEventListener("click", () => selectLevel(button.dataset.level));
}

function selectCourse(courseId, scroll = true) {
  const selectedCard = document.querySelector(`button.lesson-card[data-workspace="${courseId}"]`)
    || availableCards[0];
  if (!selectedCard) return;
  const selectedId = selectedCard.dataset.workspace;
  for (const peer of availableCards) peer.classList.toggle("selected", peer === selectedCard);
  for (const workspace of workspaces) workspace.hidden = workspace.dataset.workspacePanel !== selectedId;
  progressState.selectedCourse = selectedId;
  saveProgress();
  if (scroll) {
    document.querySelector(`[data-workspace-panel="${selectedId}"]`)
      .scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

for (const card of availableCards) {
  card.addEventListener("click", () => selectCourse(card.dataset.workspace));
}

for (const workspace of workspaces) {
  const viewButtons = workspace.querySelectorAll("[data-view]");
  const views = workspace.querySelectorAll(".view");
  for (const button of viewButtons) {
    button.addEventListener("click", () => {
      for (const peer of viewButtons) peer.setAttribute("aria-selected", String(peer === button));
      for (const view of views) view.hidden = view.id !== button.dataset.view;
    });
  }
}

function showAnswerFeedback(form) {
  for (const fieldset of form.querySelectorAll("fieldset")) {
    fieldset.querySelector(".answer-feedback")?.remove();
    const selected = fieldset.querySelector("input:checked");
    const correct = fieldset.querySelector('input[value="1"]');
    const correctText = correct?.closest("label")?.textContent.trim() || "Review the course explanation.";
    const feedback = document.createElement("p");
    const isCorrect = selected?.value === "1";
    feedback.className = `answer-feedback ${isCorrect ? "correct" : "review"}`;
    feedback.textContent = `${isCorrect ? "Correct" : "Review"} — ${correctText}`;
    fieldset.append(feedback);
  }
}

for (const form of document.querySelectorAll(".quiz-form")) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const names = [...form.querySelectorAll("fieldset input")]
      .map((input) => input.name)
      .filter((name, index, all) => all.indexOf(name) === index);
    const answered = names.map((name) => data.get(name));
    const result = form.querySelector(".quiz-result");
    if (answered.includes(null)) {
      result.textContent = "Answer every question before checking your score.";
      return;
    }
    const score = answered.reduce((total, value) => total + Number(value), 0);
    showAnswerFeedback(form);
    result.textContent = score === names.length
      ? `${score}/${names.length} — Checkpoint complete. Extend the lab next.`
      : `${score}/${names.length} — Revisit ${form.dataset.review}, then try again.`;
    if (score === names.length) {
      const courseId = form.closest("[data-workspace-panel]")?.dataset.workspacePanel;
      if (courseId) progressState.completed.add(courseId);
      saveProgress();
      updateProgress();
    }
  });
}

resetProgress.addEventListener("click", () => {
  progressState.completed.clear();
  saveProgress();
  updateProgress();
});

selectLevel(progressState.selectedLevel);
selectCourse(progressState.selectedCourse, false);
updateProgress();
