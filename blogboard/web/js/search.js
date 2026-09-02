/**
 * search.js — Global full-text search across all published articles.
 * Pre-fetches every category's articles.json, then filters by title,
 * description, and tags client-side.
 */

document.addEventListener('DOMContentLoaded', initSearch);

function initSearch() {
    const input = document.getElementById('globalSearchInput');
    const resultsEl = document.getElementById('searchResults');
    const countEl = document.getElementById('searchCount');
    if (!input || !resultsEl) return;

    let cache = null;

    async function getAllArticles() {
        if (cache) return cache;
        const all = await Promise.all(ALL_CATEGORIES.map(loadCategoryArticles));
        cache = all.flat();
        return cache;
    }

    async function runSearch() {
        const q = input.value.toLowerCase().trim();
        const articles = await getAllArticles();

        let results = articles;
        if (q) {
            results = articles.filter(a =>
                (a.title || '').toLowerCase().includes(q) ||
                (a.description || '').toLowerCase().includes(q) ||
                (a.topic || '').toLowerCase().includes(q) ||
                (a.tags || []).some(t => t.toLowerCase().includes(q))
            );
        }

        if (countEl) {
            countEl.textContent = q
                ? `${results.length} result${results.length === 1 ? '' : 's'} for "${escapeHtml(input.value.trim())}"`
                : `${results.length} articles`;
        }

        if (results.length === 0) {
            resultsEl.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <h3>No matching articles</h3>
                    <p>Try a different keyword — or browse the categories above.</p>
                </div>`;
            return;
        }

        results.sort((a, b) => new Date(b.date) - new Date(a.date));
        resultsEl.innerHTML = results.map(a => {
            const meta = CATEGORY_META[a.category] || { shortLabel: a.category, bgColor: 'transparent', color: 'inherit' };
            return `
    <a href="post.html#id=${encodeURIComponent(a.id)}" class="blog-item">
      <div class="blog-item-body">
        <div class="blog-item-meta">
          <span class="recent-cat-badge" style="background:${meta.bgColor};color:${meta.color}">${meta.shortLabel}</span>
          <span class="blog-item-date">${formatDate(a.date)}</span>
        </div>
        <h2 class="blog-item-title">${escapeHtml(a.title)}</h2>
        <p class="blog-item-desc">${escapeHtml(a.description)}</p>
        <div class="blog-item-footer"><span class="read-time">📖 ${a.readTime} read</span></div>
      </div>
      <span class="blog-item-arrow">→</span>
    </a>`;
        }).join('');
    }

    input.addEventListener('input', runSearch);
    // Support deep links: search.html#q=transformers
    const initial = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('q');
    if (initial) input.value = initial;
    runSearch();
}
