const levelButtons = document.querySelectorAll("[data-level][role='tab']");
const cards = document.querySelectorAll(".lesson-card");

for (const button of levelButtons) {
  button.addEventListener("click", () => {
    for (const peer of levelButtons) peer.setAttribute("aria-selected", String(peer === button));
    const selected = button.dataset.level;
    for (const card of cards) card.hidden = selected !== "all" && card.dataset.level !== selected;
  });
}

const viewButtons = document.querySelectorAll("[data-view]");
const views = document.querySelectorAll(".view");
for (const button of viewButtons) {
  button.addEventListener("click", () => {
    for (const peer of viewButtons) peer.setAttribute("aria-selected", String(peer === button));
    for (const view of views) view.hidden = view.id !== button.dataset.view;
  });
}

document.querySelector("[data-lesson='fundamentals']").addEventListener("click", () => {
  document.querySelector("#workspace").scrollIntoView({ behavior:"smooth", block:"start" });
});

document.querySelector("#quiz-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const answered = [data.get("q1"), data.get("q2")];
  const result = document.querySelector("#quiz-result");
  if (answered.includes(null)) {
    result.textContent = "Answer both questions before checking your score.";
    return;
  }
  const score = answered.reduce((total,value) => total + Number(value),0);
  result.textContent = score === 2 ? "2/2 — Ready to extend the lab." : `${score}/2 — Revisit the input contract and illumination-shift sections, then try again.`;
});
