/**
 * blogs-data.js — Blog data layer (bulletproof edition)
 *
 * Data resolution order:
 *   1. window.SITE_DATA (baked by scripts/build_site.js — works offline,
 *      over file://, on any static host, with zero fetches)
 *   2. fetch() of blogs/{cat}/articles.json (fallback for R2-hosted setups)
 *
 * Markdown content resolution order (used by post.js):
 *   1. window.SITE_CONTENT[id]
 *   2. fetch() of the .md file
 */

/* ── Category Metadata ─────────────────────────────────── */
const CATEGORY_META = {
  ml: {
    label: 'Machine Learning',
    shortLabel: 'ML',
    description: 'Algorithms, theory, and applied ML from fundamentals to production.',
    icon: '🧠',
    color: '#7c6af7',
    bgColor: 'rgba(124, 106, 247, 0.12)',
  },
  dl: {
    label: 'Deep Learning',
    shortLabel: 'DL',
    description: 'Neural networks, architectures, training tricks, and modern DL research.',
    icon: '🔬',
    color: '#4fc8b8',
    bgColor: 'rgba(79, 200, 184, 0.12)',
  },
  nlp: {
    label: 'Natural Language Processing',
    shortLabel: 'NLP',
    description: 'Text processing, transformers, LLMs, and language understanding.',
    icon: '📝',
    color: '#e879a0',
    bgColor: 'rgba(232, 121, 160, 0.12)',
  },
  cv: {
    label: 'Computer Vision',
    shortLabel: 'CV',
    description: 'Image processing, object detection, segmentation, and visual AI.',
    icon: '👁️',
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.12)',
  },
  genai: {
    label: 'Generative AI',
    shortLabel: 'Gen AI',
    description: 'Diffusion models, LLMs, RAG, agents, and the frontier of AI generation.',
    icon: '✨',
    color: '#a78bfa',
    bgColor: 'rgba(167, 139, 250, 0.12)',
  },
  ainews: {
    label: 'AI News',
    shortLabel: 'AI News',
    description: 'Breaking developments, model releases, and industry analysis.',
    icon: '📡',
    color: '#34d399',
    bgColor: 'rgba(52, 211, 153, 0.12)',
  },
  statistics: {
    label: 'Statistics for AI',
    shortLabel: 'Stats',
    description: 'Probability, statistical tests, distributions, and the math behind ML.',
    icon: '📊',
    color: '#fb923c',
    bgColor: 'rgba(251, 146, 60, 0.12)',
  },
};

const ALL_CATEGORIES = ['ml', 'dl', 'nlp', 'cv', 'genai', 'ainews', 'statistics'];

/* ── Cache ─────────────────────────────────────────────── */
const _cache = {};

const R2_PUBLIC_URL = (typeof window !== 'undefined' && window.CONFIG && window.CONFIG.R2_PUBLIC_URL)
  ? window.CONFIG.R2_PUBLIC_URL.replace(/\/+$/, '')
  : '';

/* ── Embedded data access ──────────────────────────────── */
function _embeddedArticles(cat) {
  if (typeof window !== 'undefined' && window.SITE_DATA && window.SITE_DATA.categories) {
    const list = window.SITE_DATA.categories[cat];
    if (Array.isArray(list)) return list;
  }
  return null;
}

/* ── Core Fetch (with embedded-first strategy) ──────────── */
async function loadCategoryArticles(cat) {
  if (_cache[cat] !== undefined) return _cache[cat];

  // 1) Baked-in data (always available — no network needed)
  const embedded = _embeddedArticles(cat);
  if (embedded) {
    _cache[cat] = embedded;
    return embedded;
  }

  // 2) Fallback: fetch registries from R2 / same-origin
  const base = R2_PUBLIC_URL || '.';
  try {
    const res = await fetch(`${base}/blogs/${cat}/articles.json`);
    if (res.ok) {
      const data = await res.json();
      _cache[cat] = Array.isArray(data) ? data : [];
      return _cache[cat];
    }
  } catch (_) { /* network unavailable — fall through */ }
  _cache[cat] = [];
  return [];
}

/* ── Markdown content loader ────────────────────────────── */
async function loadArticleContent(id, file) {
  // 1) Baked-in content
  if (typeof window !== 'undefined' && window.SITE_CONTENT && window.SITE_CONTENT[id]) {
    return window.SITE_CONTENT[id];
  }
  // 2) Fallback: fetch the .md
  const base = R2_PUBLIC_URL || '.';
  const res = await fetch(`${base}/${file}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

/* ── Queries ────────────────────────────────────────────── */
async function getBlogsByCategory(cat, sort = 'newest') {
  const articles = await loadCategoryArticles(cat);
  const sorted = [...articles].sort((a, b) =>
    sort === 'newest'
      ? new Date(b.date) - new Date(a.date)
      : new Date(a.date) - new Date(b.date)
  );
  return sorted;
}

async function getBlogById(id) {
  const parts = id.split('/');
  if (parts.length >= 3) {
    const cat = parts[1];
    if (CATEGORY_META[cat]) {
      const articles = await loadCategoryArticles(cat);
      const found = articles.find(a => a.id === id || a.file === id);
      if (found) return found;
    }
  }
  for (const cat of Object.keys(CATEGORY_META)) {
    const articles = await loadCategoryArticles(cat);
    const found = articles.find(a => a.id === id || a.file === id);
    if (found) return found;
  }
  return null;
}

async function getRecentBlogs(n = 6) {
  const all = await Promise.all(ALL_CATEGORIES.map(loadCategoryArticles));
  const flat = all.flat();
  flat.sort((a, b) => new Date(b.date) - new Date(a.date));
  return flat.slice(0, n);
}

async function getTotalCount() {
  const all = await Promise.all(ALL_CATEGORIES.map(loadCategoryArticles));
  return all.reduce((sum, arr) => sum + arr.length, 0);
}

/* ── Formatting ─────────────────────────────────────────── */
function formatDate(dateStr) {
  if (!dateStr) return '';
  const [y, m, d] = dateStr.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
  });
}
