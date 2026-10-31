/* Perfume Semantic Search: front end (plain JavaScript, no build step). */
(() => {
  "use strict";

  const REPO_URL = "https://github.com/anaskasasbeh/perfume-semantic-search";
  const PAGE = 24;
  const FAMILY_COLOR = {
    amber: "var(--f-amber)", woods: "var(--f-woods)", smoke: "var(--f-smoke)", spice: "var(--f-spice)",
    floral: "var(--f-floral)", citrus: "var(--f-citrus)", fruit: "var(--f-fruit)", fresh: "var(--f-fresh)",
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key === "style") node.setAttribute("style", value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) if (child) node.append(child);
    return node;
  };

  const state = {
    lang: "en",
    q: "",
    brand: "",
    notes: {},            // note id → "in" | "out"
    view: "home",         // "home" | "search" | "group"
    groupId: null,
    offset: 0,
    total: 0,
    catalogSize: 2191,
    filters: { brands: [], notes: [], families: [] },
    notesById: {},
    groups: [],
    examples: [],
    lastFocus: null,
  };

  /* ---------- language ---------- */
  const t = (key, vars = {}) => {
    const dict = window.I18N[state.lang] || window.I18N.en;
    let text = dict[key] ?? window.I18N.en[key] ?? key;
    for (const [name, value] of Object.entries(vars)) text = text.replaceAll(`{${name}}`, value);
    return text;
  };
  const formatNumber = (n) => new Intl.NumberFormat("en-US").format(n); // Western digits in both languages
  const noteLabel = (id) => {
    const note = state.notesById[id];
    return note ? note[state.lang] || note.en : id;
  };

  function savedLang() {
    try { return localStorage.getItem("lang"); } catch { return null; }
  }
  function saveLang(lang) {
    try { localStorage.setItem("lang", lang); } catch { /* storage unavailable */ }
  }

  function applyLang() {
    const html = document.documentElement;
    html.lang = state.lang;
    html.dir = state.lang === "ar" ? "rtl" : "ltr";
    document.title = t("brand");
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
    document.querySelectorAll("[data-i18n-aria]").forEach((node) => { node.setAttribute("aria-label", t(node.dataset.i18nAria)); });
    $("[data-i18n='heroSub']").textContent = t("heroSub", { count: formatNumber(state.catalogSize) });
    const toggle = $("#langToggle");
    toggle.textContent = t("langButton");
    toggle.setAttribute("aria-label", t("langAria"));
    renderExamples();
    renderBrandOptions();
    renderNotesPanel();
    renderChosen();
    renderShelves();
    if (state.view !== "home") rerenderResultsTitle();
  }

  /* ---------- API ---------- */
  async function api(path, params = {}) {
    const url = new URL(path, window.location.origin);
    for (const [key, value] of Object.entries(params)) {
      if (value !== "" && value !== undefined && value !== null) url.searchParams.set(key, value);
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    return response.json();
  }

  /* ---------- tiles ---------- */
  function familyStrip(noteIds) {
    const counts = {};
    for (const id of noteIds || []) {
      const family = state.notesById[id]?.family;
      if (family) counts[family] = (counts[family] || 0) + 1;
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const strip = el("span", { class: "strip", "aria-hidden": "true" });
    for (const [family, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
      strip.append(el("span", { style: `width:${(count / total) * 100}%;background:${FAMILY_COLOR[family]}` }));
    }
    return strip;
  }

  function productImage(url, alt, id) {
    const box = el("span", { class: "tile-image" });
    const placeholder = () => el("span", { class: "no-image", text: t("noImage") });
    if (!url) { box.append(placeholder()); return box; }
    // Try the retailer's link first; if the browser refuses it, ask our API to fetch it.
    const img = el("img", { src: url, alt, loading: "lazy", decoding: "async", referrerpolicy: "no-referrer" });
    img.addEventListener("error", () => {
      if (!img.dataset.proxied && id !== undefined) {
        img.dataset.proxied = "1";
        img.src = `/api/image/${id}`;
      } else {
        img.replaceWith(placeholder());
      }
    });
    box.append(img);
    return box;
  }

  function tile(card, topScore) {
    const button = el("button", { type: "button", class: "tile-button", onclick: () => openPerfume(card.id) },
      productImage(card.image_url, "", card.id),
      familyStrip(card.note_ids),
      el("p", { class: "tile-name", text: card.name }),
      el("p", { class: "tile-brand", text: card.brand }),
    );
    if (typeof card.score === "number" && topScore > 0) {
      const width = Math.max(4, Math.min(100, (card.score / topScore) * 100));
      button.append(el("span", { class: "tile-match", title: `${t("match")}: ${card.score.toFixed(3)}` },
        el("span", { text: t("match") }), el("i", { style: `--w:${width}%` })));
    }
    button.setAttribute("aria-label", `${card.name}, ${card.brand}`);
    return el("li", { class: "tile" }, button);
  }

  /* ---------- examples, brands, notes ---------- */
  function renderExamples() {
    const box = $("#examples");
    box.replaceChildren(el("span", { class: "examples-label", text: t("examplesLabel") }));
    for (const example of state.examples.filter((e) => e.lang === state.lang)) {
      box.append(el("button", { type: "button", class: "example", text: example.text,
        onclick: () => { $("#q").value = example.text; startSearch(); } }));
    }
  }

  function renderBrandOptions() {
    const select = $("#brand");
    const current = state.brand;
    select.replaceChildren(el("option", { value: "", text: t("allBrands") }));
    for (const brand of state.filters.brands) {
      select.append(el("option", { value: brand.name, text: `${brand.name} (${formatNumber(brand.count)})` }));
    }
    select.value = current;
  }

  function renderNotesPanel() {
    const box = $("#notesGroups");
    box.replaceChildren();
    for (const family of state.filters.families) {
      const notes = state.filters.notes.filter((n) => n.family === family.id);
      if (!notes.length) continue;
      const list = el("div", { class: "family-notes" });
      for (const note of notes) {
        list.append(el("button", {
          type: "button", class: "note-toggle", "data-state": state.notes[note.id] || "off",
          "aria-pressed": state.notes[note.id] ? "true" : "false",
          onclick: () => cycleNote(note.id),
        }, note[state.lang] || note.en, el("small", { text: formatNumber(note.count) })));
      }
      box.append(el("div", { class: "family" },
        el("h3", {}, el("span", { class: "dot", style: `background:${FAMILY_COLOR[family.id]}` }), family[state.lang] || family.en),
        list));
    }
  }

  function renderChosen() {
    const box = $("#chosenNotes");
    box.replaceChildren();
    const entries = Object.entries(state.notes);
    for (const [id, mode] of entries) {
      const family = state.notesById[id]?.family;
      box.append(el("span", { class: `chip ${mode === "out" ? "is-out" : ""}` },
        el("span", { class: "dot", style: `background:${FAMILY_COLOR[family] || "var(--f-none)"}` }),
        `${t(mode === "in" ? "include" : "exclude")} ${noteLabel(id)}`,
        el("button", { type: "button", class: "chip-x", "aria-label": t("removeNote", { note: noteLabel(id) }),
          onclick: () => { delete state.notes[id]; filtersChanged(); } }, "×")));
    }
    $("#notesCount").textContent = entries.length ? formatNumber(entries.length) : "";
    $("#clearAll").hidden = !(entries.length || state.brand || state.q);
  }

  function cycleNote(id) {
    const next = { off: "in", in: "out", out: "off" }[state.notes[id] || "off"];
    if (next === "off") delete state.notes[id]; else state.notes[id] = next;
    renderNotesPanel();
    filtersChanged();
  }

  function filtersChanged() {
    renderChosen();
    if (state.view === "group") state.view = state.q ? "search" : "home";
    const hasFilters = state.brand || Object.keys(state.notes).length;
    if (state.q || hasFilters) runSearch(true); else showHome();
  }

  /* ---------- shelves (home) ---------- */
  function groupTitleParts(group) {
    if (!group.note_ids || !group.note_ids.length) return [state.lang === "ar" ? group.name_ar : group.name_en];
    return group.note_ids.map((id) => {
      const family = state.notesById[id]?.family;
      return el("span", { class: "note-word" },
        el("span", { class: "dot", style: `background:${FAMILY_COLOR[family] || "var(--f-none)"}`, "aria-hidden": "true" }),
        noteLabel(id));
    });
  }

  function groupTitle(group) {
    return el("h3", { class: "shelf-title" }, groupTitleParts(group));
  }

  function renderShelves() {
    const list = $("#shelfList");
    list.replaceChildren();
    for (const group of state.groups) {
      const track = el("ol", { class: "shelf-track" }, group.items.map((card) => tile(card, 0)));
      const scroll = (dir) => {
        const sign = document.documentElement.dir === "rtl" ? -1 : 1;
        track.scrollBy({ left: sign * dir * track.clientWidth * 0.8, behavior: "smooth" });
      };
      list.append(el("section", { class: "shelf" },
        el("div", { class: "shelf-head" },
          groupTitle(group),
          el("span", { class: "shelf-meta", text: t("countPerfumes", { n: formatNumber(group.size) }) }),
          el("button", { type: "button", class: "link", text: t("seeAll", { n: formatNumber(group.size) }),
            onclick: () => showGroup(group.id) }),
          el("span", { class: "shelf-nav" },
            el("button", { type: "button", "aria-label": t("scrollPrev"), onclick: () => scroll(-1) }, "‹"),
            el("button", { type: "button", "aria-label": t("scrollNext"), onclick: () => scroll(1) }, "›"))),
        track));
    }
  }

  /* ---------- results ---------- */
  function showHome() {
    state.view = "home";
    $("#results").hidden = true;
    $("#shelves").hidden = false;
    syncUrl();
  }

  function showResultsSection() {
    $("#results").hidden = false;
    $("#shelves").hidden = true;
  }

  function rerenderResultsTitle() {
    const title = $("#resultsTitle");
    if (state.view === "group") {
      const group = state.groups.find((g) => g.id === state.groupId);
      title.className = "shelf-title";
      title.replaceChildren(...(group ? groupTitleParts(group) : []));
    } else {
      title.className = "";
      title.textContent = state.q ? t("resultsFor", { q: state.q }) : t("resultsFiltered");
    }
    $("#backToShelves").textContent = t("backToShelves");
  }

  function setBusy(busy) {
    const form = $("#searchForm");
    form.classList.toggle("is-busy", busy);
    form.classList.toggle("is-done", !busy);
    $("#results").setAttribute("aria-busy", busy ? "true" : "false");
  }

  function renderNotice(result) {
    const notice = $("#notice");
    const key = result.notice === "fallback" ? "noticeFallback" : result.notice === "arabic_unavailable" ? "noticeArabic" : null;
    notice.hidden = !key;
    if (key) notice.textContent = t(key);
  }

  function renderMeta(key) {
    $("#resultsMeta").textContent = key && state.total ? t(key, { n: formatNumber(state.total) }) : "";
  }

  function revealResults() {
    const box = $("#results");
    if (box.getBoundingClientRect().top > window.innerHeight * 0.6) box.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderResults(items, append, emptyKey) {
    const grid = $("#resultsGrid");
    if (!append) grid.replaceChildren();
    const top = items.length && typeof items[0].score === "number" ? (append ? state.topScore : items[0].score) : 0;
    if (!append) state.topScore = top;
    for (const card of items) grid.append(tile(card, state.topScore));
    if (!grid.children.length && emptyKey) grid.append(el("li", { class: "empty", text: t(emptyKey) }));
    $("#loadMore").hidden = state.offset >= state.total;
  }

  async function runSearch(reset, userAction = true) {
    if (reset) state.offset = 0;
    state.view = "search";
    showResultsSection();
    rerenderResultsTitle();
    setBusy(true);
    syncUrl();
    try {
      const include = Object.keys(state.notes).filter((id) => state.notes[id] === "in").join(",");
      const exclude = Object.keys(state.notes).filter((id) => state.notes[id] === "out").join(",");
      const result = await api("/api/search", { q: state.q, brand: state.brand, include, exclude, limit: PAGE, offset: state.offset });
      state.total = result.total;
      state.offset += result.results.length;
      renderNotice(result);
      renderMeta({ semantic: "metaSemantic", keyword: "metaKeyword", browse: "metaBrowse" }[result.mode]);
      const emptyKey = result.notice === "arabic_unavailable" ? null
        : state.q && !(Object.keys(state.notes).length || state.brand) ? "emptyQuery" : "empty";
      renderResults(result.results, !reset, emptyKey);
      if (reset && userAction) revealResults();
    } catch (error) {
      console.error(error);
      $("#notice").hidden = false;
      $("#notice").textContent = t("error");
    } finally {
      setBusy(false);
    }
  }

  async function showGroup(groupId, reset = true) {
    state.view = "group";
    state.groupId = groupId;
    if (reset) state.offset = 0;
    showResultsSection();
    rerenderResultsTitle();
    $("#notice").hidden = true;
    const data = await api(`/api/groups/${groupId}`, { offset: state.offset, limit: PAGE });
    state.total = data.size;
    state.offset += data.items.length;
    renderMeta("metaGroup");
    renderResults(data.items, !reset, "empty");
    if (reset) $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function startSearch() {
    state.q = $("#q").value.trim();
    renderChosen();
    const hasFilters = state.brand || Object.keys(state.notes).length;
    if (!state.q && !hasFilters) { showHome(); return; }
    runSearch(true);
  }

  /* ---------- detail drawer ---------- */
  async function openPerfume(id) {
    state.lastFocus = document.activeElement;
    const body = $("#drawerBody");
    body.replaceChildren(el("p", { text: t("loading") }));
    $("#drawer").hidden = false;
    $("#scrim").hidden = false;
    document.body.style.overflow = "hidden";
    $("#drawerClose").focus();
    try {
      const [perfume, similar] = await Promise.all([api(`/api/perfumes/${id}`), api(`/api/perfumes/${id}/similar`, { limit: 10 })]);
      const keyNotes = el("div", { class: "key-notes" }, perfume.note_ids.map((noteId) => el("span", { class: "chip" },
        el("span", { class: "dot", style: `background:${FAMILY_COLOR[state.notesById[noteId]?.family] || "var(--f-none)"}` }),
        noteLabel(noteId))));
      const topScore = similar.length ? similar[0].score : 0;
      body.replaceChildren(el("article", { class: "detail" },
        el("div", { class: "detail-image" }, productImage(perfume.image_url, perfume.name, perfume.id).firstChild || ""),
        el("h2", { class: "detail-name", id: "drawerName", text: perfume.name }),
        el("p", { class: "detail-brand", text: t("by", { brand: perfume.brand }) }),
        perfume.note_ids.length ? el("h3", { text: t("keyNotes") }) : null,
        perfume.note_ids.length ? keyNotes : null,
        perfume.notes.length ? el("h3", { text: t("allNotes") }) : null,
        perfume.notes.length ? el("p", { class: "raw-notes", text: perfume.notes.join(", ") }) : null,
        perfume.description ? el("h3", { text: t("about") }) : null,
        perfume.description ? el("p", { class: "description", text: perfume.description }) : null,
        el("h3", { text: t("similar") }),
        el("ol", { class: "shelf-track" }, similar.map((card) => tile(card, topScore)))));
      $("#drawer").scrollTop = 0;
    } catch (error) {
      console.error(error);
      body.replaceChildren(el("p", { text: t("error") }));
    }
  }

  function closeDrawer() {
    $("#drawer").hidden = true;
    $("#scrim").hidden = true;
    document.body.style.overflow = "";
    if (state.lastFocus && document.contains(state.lastFocus)) state.lastFocus.focus();
  }

  /* ---------- URL state (shareable links) ---------- */
  function syncUrl() {
    const params = new URLSearchParams();
    if (state.q) params.set("q", state.q);
    if (state.brand) params.set("brand", state.brand);
    const include = Object.keys(state.notes).filter((id) => state.notes[id] === "in");
    const exclude = Object.keys(state.notes).filter((id) => state.notes[id] === "out");
    if (include.length) params.set("in", include.join(","));
    if (exclude.length) params.set("out", exclude.join(","));
    if (state.lang === "ar") params.set("lang", "ar");
    const query = params.toString();
    history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
  }

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    state.q = params.get("q") || "";
    state.brand = params.get("brand") || "";
    for (const id of (params.get("in") || "").split(",").filter(Boolean)) state.notes[id] = "in";
    for (const id of (params.get("out") || "").split(",").filter(Boolean)) state.notes[id] = "out";
    const lang = params.get("lang") || savedLang() || (navigator.language || "").slice(0, 2);
    state.lang = lang === "ar" ? "ar" : "en";
  }

  /* ---------- wiring ---------- */
  function bindEvents() {
    $("#searchForm").addEventListener("submit", (event) => { event.preventDefault(); startSearch(); });
    $("#brand").addEventListener("change", (event) => { state.brand = event.target.value; filtersChanged(); });
    $("#langToggle").addEventListener("click", () => {
      state.lang = state.lang === "ar" ? "en" : "ar";
      saveLang(state.lang);
      applyLang();
      syncUrl();
    });
    const panel = $("#notesPanel");
    const button = $("#notesButton");
    const togglePanel = (open) => { panel.hidden = !open; button.setAttribute("aria-expanded", String(open)); };
    button.addEventListener("click", () => togglePanel(panel.hidden));
    $("#notesDone").addEventListener("click", () => { togglePanel(false); button.focus(); });
    $("#clearAll").addEventListener("click", () => {
      state.notes = {}; state.brand = ""; state.q = ""; $("#q").value = ""; $("#brand").value = "";
      renderNotesPanel(); renderChosen(); showHome();
    });
    $("#backToShelves").addEventListener("click", () => {
      state.q = ""; $("#q").value = ""; state.notes = {}; state.brand = ""; $("#brand").value = "";
      renderNotesPanel(); renderChosen(); showHome();
      $("#shelves").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    $("#loadMore").addEventListener("click", () => (state.view === "group" ? showGroup(state.groupId, false) : runSearch(false)));
    $("#drawerClose").addEventListener("click", closeDrawer);
    $("#scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !$("#drawer").hidden) closeDrawer();
      if (event.key === "Escape" && !panel.hidden) togglePanel(false);
    });
  }

  async function init() {
    readUrl();
    bindEvents();
    if (REPO_URL) { $("#repoLink").href = REPO_URL; $("#repoLink").hidden = false; }
    try {
      const [filters, groups, examples, health] = await Promise.all([
        api("/api/filters"), api("/api/groups", { per_group: 16 }), api("/api/examples"), api("/api/health"),
      ]);
      state.filters = filters;
      state.notesById = Object.fromEntries(filters.notes.map((n) => [n.id, n]));
      state.groups = groups;
      state.examples = examples;
      state.catalogSize = health.perfumes;
    } catch (error) {
      console.error(error);
    }
    $("#q").value = state.q;
    applyLang();
    if (state.q || state.brand || Object.keys(state.notes).length) runSearch(true, false);
  }

  init();
})();
