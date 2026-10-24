/**
 * article.js — renders a statically generated article page.
 * The page injects window.ARTICLE_ID at build time; this script pulls
 * metadata + content from SITE_DATA / SITE_CONTENT and renders everything.
 */

document.addEventListener('DOMContentLoaded', async () => {
    initNavBurger();
    initThemeToggle();
    initReadingProgress();
    await loadArticlePage();
});

function initNavBurger() {
    const navbar = document.getElementById('navbar');
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 30);
    });
    hamburger?.addEventListener('click', () => {
        hamburger.classList.toggle('open');
        navLinks.classList.toggle('open');
    });
}

function initReadingProgress() {
    const bar = document.getElementById('readingProgress');
    if (!bar) return;
    window.addEventListener('scroll', () => {
        const doc = document.documentElement;
        const scrollTop = doc.scrollTop || document.body.scrollTop;
        const scrollHeight = doc.scrollHeight - doc.clientHeight;
        bar.style.width = scrollHeight > 0 ? `${(scrollTop / scrollHeight) * 100}%` : '0%';
    });
}

async function loadArticlePage() {
    const id = window.ARTICLE_ID;
    const contentEl = document.getElementById('postContent');

    const blog = await getBlogById(id);
    if (!blog) {
        showError('Article not found.', contentEl);
        return;
    }

    const meta = CATEGORY_META[blog.category] || {
        label: blog.category, shortLabel: blog.category,
        bgColor: 'transparent', color: 'inherit',
    };

    document.title = `${blog.title} — NeuraPress`;

    const catLink = document.getElementById('catLink');
    if (catLink) {
        catLink.textContent = meta.label;
        catLink.href = `../category.html#cat=${blog.category}`;
    }
    const postTitleSpan = document.getElementById('postTitle');
    if (postTitleSpan) postTitleSpan.textContent = blog.title;

    document.getElementById('postTitleH1').textContent = blog.title;
    document.getElementById('postDate').textContent = formatDate(blog.date);

    const catBadge = document.getElementById('postCatBadge');
    if (catBadge) {
        catBadge.textContent = meta.shortLabel;
        catBadge.className = `post-cat-badge badge-${blog.category}`;
    }
    const readTimeEl = document.getElementById('postReadTime');
    if (readTimeEl) readTimeEl.textContent = `📖 ${blog.readTime} read`;

    document.querySelectorAll('.nav-link[href]').forEach(link => {
        if (link.href.includes(`cat=${blog.category}`)) link.classList.add('active');
    });

    const backBtn = document.getElementById('backToCat');
    if (backBtn) {
        backBtn.href = `../category.html#cat=${blog.category}`;
        backBtn.textContent = `← Back to ${meta.shortLabel}`;
    }

    const tagsEl = document.getElementById('postTags');
    if (tagsEl && blog.tags?.length) {
        tagsEl.innerHTML = blog.tags.map(t => `<span class="post-tag">#${escapeHtml(t)}</span>`).join('');
    }

    const coverEl = document.getElementById('postCover');
    if (coverEl && blog.coverImage) {
        coverEl.innerHTML = `<img src="${blog.coverImage}" alt="" loading="lazy">`;
        coverEl.style.display = 'block';
    }

    try {
        const mdText = await loadArticleContent(blog.id, blog.file);
        renderMarkdown(mdText, contentEl);
        buildTOC();
        await loadRelatedPosts(blog);
    } catch (err) {
        showError('Could not load the article content.', contentEl);
        console.error('Failed to load article:', err);
    }
}

function renderMarkdown(mdText, container) {
    if (typeof marked === 'undefined') {
        const safe = mdText
            .replace(/```[\s\S]*?```/g, m => m.replace(/```(\w*)\n?/g, ''))
            .replace(/^#{1,6}\s+/gm, '')
            .replace(/\*\*([^*]+)\*\*/g, '$1');
        container.innerHTML = `<div style="white-space:pre-wrap">${escapeHtml(safe)}</div>`;
        return;
    }
    marked.setOptions({ gfm: true, breaks: true });

    const renderer = new marked.Renderer();
    renderer.heading = (text, level) => {
        const rawText = typeof text === 'object' ? text.text : text;
        const escapedText = rawText.toLowerCase().replace(/[^\w]+/g, '-');
        return `<h${level} id="${escapedText}">${rawText}</h${level}>`;
    };
    container.innerHTML = marked.parse(mdText, { renderer });

    if (window.hljs) {
        container.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }
    container.querySelectorAll('pre').forEach(pre => {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        btn.style.cssText = `position:absolute; top:10px; right:12px; background:rgba(124,106,247,0.15); color:#a89cf7; border:1px solid rgba(124,106,247,0.25); border-radius:6px; padding:3px 10px; font-size:0.75rem; cursor:pointer; font-family:var(--font-sans);`;
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code?.textContent || '').then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => (btn.textContent = 'Copy'), 2000);
            });
        });
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}

function buildTOC() {
    const content = document.getElementById('postContent');
    const tocNav = document.getElementById('tocNav');
    if (!content || !tocNav) return;
    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) return;
    tocNav.innerHTML = '';
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                tocNav.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
                tocNav.querySelector(`[data-target="${entry.target.id}"]`)?.classList.add('active');
            }
        });
    }, { rootMargin: '-20% 0px -70% 0px' });
    headings.forEach(h => {
        const link = document.createElement('a');
        link.href = `#${h.id}`;
        link.setAttribute('data-target', h.id);
        link.textContent = h.textContent;
        link.className = `toc-link level-${h.tagName.toLowerCase()}`;
        link.addEventListener('click', e => {
            e.preventDefault();
            document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        tocNav.appendChild(link);
        observer.observe(h);
    });
}

async function loadRelatedPosts(currentBlog, n = 3) {
    const wrap = document.getElementById('relatedPosts');
    if (!wrap) return;
    const currentTags = currentBlog.tags || [];
    const pool = [];
    const catArticles = await loadCategoryArticles(currentBlog.category);
    for (const a of catArticles) {
        if (a.id !== currentBlog.id) pool.push({ article: a, score: 2 });
    }
    for (const cat of Object.keys(CATEGORY_META)) {
        if (cat === currentBlog.category) continue;
        const arts = await loadCategoryArticles(cat);
        for (const a of arts) {
            const shared = (a.tags || []).filter(t => currentTags.includes(t)).length;
            if (shared > 0) pool.push({ article: a, score: shared });
        }
    }
    pool.sort((x, y) => y.score - x.score || new Date(y.article.date) - new Date(x.article.date));
    const seen = new Set();
    const related = [];
    for (const { article } of pool) {
        if (seen.has(article.id)) continue;
        seen.add(article.id);
        related.push(article);
        if (related.length >= n) break;
    }
    if (related.length === 0) {
        wrap.closest('.related-section')?.classList.add('hidden');
        return;
    }
    wrap.innerHTML = related.map(a => {
        const meta = CATEGORY_META[a.category] || { shortLabel: a.category, bgColor: 'transparent', color: 'inherit' };
        return `
    <a href="../post.html#id=${encodeURIComponent(a.id)}" class="recent-card related-card">
      <div class="recent-card-meta">
        <span class="recent-cat-badge" style="background:${meta.bgColor};color:${meta.color}">${meta.shortLabel}</span>
        <span class="recent-date">${formatDate(a.date)}</span>
      </div>
      <h3 class="recent-title">${escapeHtml(a.title)}</h3>
      <p class="recent-desc">${escapeHtml(a.description)}</p>
    </a>`;
    }).join('');
}

function showError(message, container) {
    if (!container) return;
    container.innerHTML = `
    <div style="padding:40px;text-align:center;color:var(--text-muted)">
      <div style="font-size:2.5rem;margin-bottom:16px">⚠️</div>
      <h3 style="color:var(--text-secondary);margin-bottom:12px">Unable to load article</h3>
      <p style="line-height:1.7">${message}</p>
    </div>`;
}
