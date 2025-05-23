// scripts.js

// Example: Add simple client-side validation or interactivity if needed

document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');

    if (form) {
        form.addEventListener('submit', (e) => {
            const horizonInput = document.getElementById('horizon');
            const horizon = Number(horizonInput.value);

            if (isNaN(horizon) || horizon < 1 || horizon > 30) {
                e.preventDefault();
                alert('Please enter a forecast horizon between 1 and 30 days.');
                horizonInput.focus();
            }
        });
    }
});
