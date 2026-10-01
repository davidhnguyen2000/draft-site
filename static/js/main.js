// Draft project page: lazy autoplay, slider fill, BibTeX copy.

// Range sliders draw their filled part from --p; keep it at the thumb, whether
// the user moved it or a script set its value.
window.paintRange = (el) => {
  const min = Number(el.min || 0), max = Number(el.max || 100);
  el.style.setProperty("--p", `${((Number(el.value) - min) / (max - min || 1)) * 100}%`);
};
document.addEventListener("input", (e) => { if (e.target.type === "range") window.paintRange(e.target); });

// Play muted loops only while they are on screen, so a dozen clips do not all
// decode at once.
const observer = new IntersectionObserver((entries) => {
  for (const { target, isIntersecting } of entries) {
    if (isIntersecting) {
      target.play().catch(() => {});
    } else {
      target.pause();
    }
  }
}, { threshold: 0.25 });

document.querySelectorAll("video.autoplay").forEach((video) => observer.observe(video));

document.querySelector(".copy").addEventListener("click", async (event) => {
  const label = event.currentTarget.querySelector("span");
  try {
    await navigator.clipboard.writeText(document.getElementById("bibtex-code").textContent);
    label.textContent = "Copied";
  } catch {
    label.textContent = "Select and copy";
  }
  setTimeout(() => { label.textContent = "Copy"; }, 1800);
});
