/**
 * main.js — Shared frontend utilities (theme toggle, nav, helpers).
 * Loaded on every page before page-specific scripts.
 */

/* ── Theme Toggle (dark ⇆ light, persisted) ─────────────── */
function initThemeToggle() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;

    const saved = localStorage.getItem('bb-theme');
    if (saved === 'light') document.body.classList.add('theme-light');
    updateThemeIcon(btn);

    btn.addEventListener('click', () => {
        document.body.classList.toggle('theme-light');
        const light = document.body.classList.contains('theme-light');
        localStorage.setItem('bb-theme', light ? 'light' : 'dark');
        updateThemeIcon(btn);
    });
}

function updateThemeIcon(btn) {
    const light = document.body.classList.contains('theme-light');
    btn.textContent = light ? '☀' : '☾';
    btn.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
}

/* ── Escape helper ──────────────────────────────────────── */
function escapeHtml(str) {
    return (str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

document.addEventListener('DOMContentLoaded', initThemeToggle);
