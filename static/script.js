// Enable responsive behavior for the Q&A section
window.addEventListener('resize', () => {
    const qaContainer = document.getElementById('qa-container');
    if (window.innerWidth < 600) {
        qaContainer.style.flexDirection = 'column';
    } else {
        qaContainer.style.flexDirection = 'row';
    }
});
