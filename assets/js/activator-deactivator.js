(() => {

  const activators = document.querySelectorAll('[data-target]');
  const targets = document.querySelectorAll('[data-id].hidden');

  function handleActivatorClick(e) {
    const activator = e.currentTarget;
    const targetId = activator.dataset.target;
    const target = document.querySelector(`[data-id="${targetId}"]`);

    // Hide all targets
    targets.forEach(target => target.classList.add("hidden"));

    // Show the selected target
    if (target) {
      target.classList.remove("hidden");
    }

  }

  // Add event listeners
  activators.forEach(activator => {
    activator.addEventListener('click', handleActivatorClick);
    // activator.removeEventListener('click', handleActivatorClick);
  });

  document.body.addEventListener("click", () => {
    targets.forEach(target => target.classList.add("hidden"));
  })
})();
