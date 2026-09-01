const levelButtons = document.querySelectorAll("[data-level][role='tab']");
const cards = document.querySelectorAll(".lesson-card");

for (const button of levelButtons) {
  button.addEventListener("click", () => {
    for (const peer of levelButtons) peer.setAttribute("aria-selected", String(peer === button));
    const selected = button.dataset.level;
    for (const card of cards) card.hidden = selected !== "all" && card.dataset.level !== selected;
  });
}

const availableCards = document.querySelectorAll("button.lesson-card[data-workspace]");
const workspaces = document.querySelectorAll("[data-workspace-panel]");

for (const card of availableCards) {
  card.addEventListener("click", () => {
    for (const peer of availableCards) peer.classList.toggle("selected", peer === card);
    for (const workspace of workspaces) {
      workspace.hidden = workspace.dataset.workspacePanel !== card.dataset.workspace;
    }
    document.querySelector(`[data-workspace-panel="${card.dataset.workspace}"]`)
      .scrollIntoView({ behavior: "smooth", block: "start" });
  });
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
    result.textContent = score === names.length
      ? `${score}/${names.length} — Ready to extend the lab.`
      : `${score}/${names.length} — Revisit ${form.dataset.review}, then try again.`;
  });
}
