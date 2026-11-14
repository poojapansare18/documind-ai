const state = { documents: [], selectedId: null, selected: null, pages: [], view: "overview" };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" && payload ? payload.detail : payload;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return payload;
}

function setStatus(selector, message = "", type = "") {
  const node = $(selector);
  node.textContent = message;
  node.className = `inline-status ${type}`.trim();
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function formatDate(value) {
  if (!value) return "Date unavailable";
  const date = new Date(value.replace(" ", "T") + "Z");
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(undefined, {year:"numeric",month:"short",day:"numeric"});
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showView(view) {
  state.view = view;
  $$(".page").forEach(page => page.classList.toggle("active", page.id === `view-${view}`));
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === view));
}

function renderStats() {
  $("#stat-documents").textContent = formatNumber(state.documents.length);
  $("#stat-pages").textContent = formatNumber(state.documents.reduce((sum, doc) => sum + doc.page_count, 0));
  $("#stat-characters").textContent = formatNumber(state.documents.reduce((sum, doc) => sum + doc.character_count, 0));
  $("#stat-selected").textContent = state.selected ? "1" : "—";
  $("#document-count").textContent = `${state.documents.length} file${state.documents.length === 1 ? "" : "s"}`;
}

function renderDocuments() {
  const list = $("#document-list");
  list.replaceChildren();
  if (!state.documents.length) {
    const empty = element("div", "empty-state");
    empty.append(element("div", "empty-symbol", "▤"), element("strong", "", "No documents yet"), element("p", "", "Upload a PDF to start building your knowledge base."));
    list.append(empty);
    return;
  }
  state.documents.forEach(doc => {
    const button = element("button", `document-item ${doc.document_id === state.selectedId ? "selected" : ""}`);
    button.type = "button";
    const icon = element("span", "file-icon", "PDF");
    const copy = element("span", "document-copy");
    copy.append(element("strong", "", doc.filename), element("span", "", `${doc.page_count} pages · ${formatNumber(doc.character_count)} characters · ${formatDate(doc.created_at)}`));
    button.append(icon, copy, element("span", "document-chevron", "›"));
    button.addEventListener("click", () => selectDocument(doc.document_id));
    list.append(button);
  });
}

async function loadDocuments() {
  state.documents = await api("/documents");
  renderStats();
  renderDocuments();
}

function renderPageMetadata() {
  const container = $("#page-metadata");
  container.replaceChildren();
  state.pages.forEach(page => {
    const chips = [
      `Page ${page.page_number}`,
      page.extraction_method === "ocr" ? "OCR processed" : "Native text",
      page.layout_type === "two_column" ? "Two columns" : "Single column",
      `${formatNumber(page.character_count)} characters`,
    ];
    chips.forEach((label, index) => {
      const chip = element("span", `meta-chip ${index === 1 && page.ocr_applied ? "ocr" : ""}`, label);
      container.append(chip);
    });
  });
}

async function selectDocument(documentId) {
  state.selectedId = documentId;
  state.selected = state.documents.find(doc => doc.document_id === documentId) || null;
  renderDocuments();
  renderStats();
  if (!state.selected) return;
  $("#selected-empty").classList.add("hidden");
  $("#selected-content").classList.remove("hidden");
  $("#selected-badge").textContent = "Selected";
  $("#selected-filename").textContent = state.selected.filename;
  $("#selected-meta").textContent = `${state.selected.page_count} pages · ${formatNumber(state.selected.character_count)} characters`;
  $("#selected-summary").textContent = "Loading summary…";
  $("#page-preview").textContent = "Loading page text…";
  $("#index-button").disabled = false;
  $("#tables-button").disabled = false;
  $("#delete-button").disabled = false;
  $("#ask-scope").textContent = state.selected.filename;
  $("#ask-scope-hint").textContent = "Questions will be limited to this selected document.";
  $("#tables-scope").textContent = state.selected.filename;
  try {
    const [summary, pages] = await Promise.all([
      api(`/documents/${documentId}/summary`),
      api(`/documents/${documentId}/pages`)
    ]);
    state.pages = pages;
    $("#selected-summary").textContent = summary.summary || "No extractive summary is available.";
    $("#page-preview").textContent = pages[0]?.text || "No text was extracted from page one.";
    renderPageMetadata();
    setStatus("#document-action-status", "", "");
  } catch (error) {
    $("#selected-summary").textContent = "Could not load document details.";
    setStatus("#document-action-status", error.message, "error");
  }
}

function renderSearchResults(results, mode) {
  const container = $("#search-results");
  container.replaceChildren();
  if (!results.length) {
    const empty = element("div", "empty-state");
    empty.append(element("div", "empty-symbol", "⌕"), element("strong", "", "No matches found"), element("p", "", "Try a different phrase or search mode."));
    container.append(empty);
    return;
  }
  results.forEach(result => {
    const card = element("article", "result-card");
    const top = element("div", "result-top");
    const title = element("strong", "", result.filename);
    const score = element("span", "result-score", mode === "semantic" ? `Similarity ${Number(result.score).toFixed(3)}` : `Score ${result.score}`);
    top.append(title, score);
    const meta = element("div", "result-meta", `Page ${result.page_number} · ${result.document_id.slice(0, 8)}`);
    const snippet = element("p", "result-snippet", result.snippet || result.text || "");
    const open = element("button", "text-button", "Open document");
    open.type = "button";
    open.addEventListener("click", () => {
      selectDocument(result.document_id).then(() => showView("overview"));
    });
    card.append(top, meta, snippet, open);
    container.append(card);
  });
}

async function runSearch(event) {
  event.preventDefault();
  const query = $("#search-query").value.trim();
  const mode = $("#search-mode").value;
  if (!query) return;
  setStatus("#search-status", mode === "semantic" ? "Searching indexed passages…" : "Searching document text…");
  $("#search-results").classList.add("loading");
  try {
    const results = mode === "semantic"
      ? await api("/documents/semantic-search", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query,limit:10})})
      : await api("/documents/search", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query,limit:10})});
    renderSearchResults(results, mode);
    setStatus("#search-status", `${results.length} result${results.length === 1 ? "" : "s"} found.`, "success");
  } catch (error) {
    setStatus("#search-status", error.message, "error");
  } finally {
    $("#search-results").classList.remove("loading");
  }
}

function renderAnswer(result) {
  const container = $("#answer-panel");
  container.replaceChildren();
  const card = element("div", "answer-card");
  card.append(element("div", "detail-label", "DOCUMENT-GROUNDED ANSWER"), element("div", "answer-text", result.answer));
  container.append(card);
  if (result.sources?.length) {
    container.append(element("div", "sources-heading", `SOURCE EXCERPTS · ${result.sources.length}`));
    result.sources.forEach(source => {
      const item = element("div", "source-item");
      item.append(element("strong", "", `[${source.source_id}] ${source.filename} · Page ${source.page_number}`));
      item.append(element("p", "", source.text));
      container.append(item);
    });
  }
}

async function runAsk(event) {
  event.preventDefault();
  const question = $("#ask-question").value.trim();
  if (!question) return;
  setStatus("#ask-status", "Retrieving evidence and generating an answer…");
  $("#answer-panel").classList.add("loading");
  try {
    const result = await api("/documents/ask", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question,limit:5,document_id:state.selectedId || null})
    });
    renderAnswer(result);
    setStatus("#ask-status", "Answer generated. Review the source excerpts before relying on it.", "success");
  } catch (error) {
    setStatus("#ask-status", error.message, "error");
  } finally {
    $("#answer-panel").classList.remove("loading");
  }
}

function renderTables(pages) {
  const container = $("#tables-results");
  container.replaceChildren();
  let tableTotal = 0;
  pages.forEach(page => {
    (page.tables || []).forEach((table, tableIndex) => {
      tableTotal += 1;
      const card = element("section", "table-card");
      const header = element("div", "table-card-header");
      header.append(element("span", "", `Page ${page.page_number} · Table ${tableIndex + 1}`));
      const detail = [
        `${table.row_count} rows`,
        `${table.column_count} columns`,
        table.detection_method || "heuristic",
        table.continues_from_previous_page ? "Continued from previous page" : "",
        table.continues_on_next_page ? "Continues on next page" : "",
      ].filter(Boolean).join(" · ");
      header.append(element("span", "result-meta", detail));
      const scroll = element("div", "table-scroll");
      const tableNode = document.createElement("table");
      (table.rows || []).forEach((row, rowIndex) => {
        const tr = document.createElement("tr");
        (row.cells || []).forEach(cell => {
          const td = document.createElement(rowIndex === 0 ? "th" : "td");
          td.textContent = cell.text || "";
          td.colSpan = Math.max(1, Number(cell.column_span || 1));
          td.rowSpan = Math.max(1, Number(cell.row_span || 1));
          tr.append(td);
        });
        tableNode.append(tr);
      });
      scroll.append(tableNode);
      card.append(header, scroll);
      container.append(card);
    });
  });
  if (!tableTotal) {
    const empty = element("div", "empty-state");
    empty.append(element("div", "empty-symbol", "▤"), element("strong", "", "No table candidates detected"), element("p", "", "This PDF may contain paragraphs, unsupported layouts or scanned tables that need additional processing."));
    container.append(empty);
  }
}

async function loadTables() {
  if (!state.selectedId) {
    setStatus("#tables-status", "Select a document in Overview first.", "error");
    return;
  }
  setStatus("#tables-status", "Inspecting page layouts…");
  $("#refresh-tables").disabled = true;
  try {
    const pages = await api(`/documents/${state.selectedId}/tables`);
    renderTables(pages);
    const count = pages.reduce((sum, page) => sum + (page.tables || []).length, 0);
    setStatus("#tables-status", `${count} table candidate${count === 1 ? "" : "s"} found across ${pages.length} pages.`, "success");
  } catch (error) {
    setStatus("#tables-status", error.message, "error");
  } finally {
    $("#refresh-tables").disabled = false;
  }
}

async function indexSelectedDocument() {
  if (!state.selectedId) return;
  const button = $("#index-button");
  button.disabled = true;
  button.textContent = "Indexing…";
  setStatus("#document-action-status", "Generating local embeddings. The first run may download the model.");
  try {
    const result = await api(`/documents/${state.selectedId}/index`, {method:"POST"});
    setStatus("#document-action-status", `${result.indexed_chunks} chunks indexed · ${result.dimensions} dimensions · ${result.embedding_model}`, "success");
  } catch (error) {
    setStatus("#document-action-status", error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Create search index";
  }
}

async function deleteSelectedDocument() {
  if (!state.selectedId || !confirm("Delete this document and its extracted data?")) return;
  const button = $("#delete-button");
  button.disabled = true;
  try {
    await api(`/documents/${state.selectedId}`, {method:"DELETE"});
    state.selectedId = null;
    state.selected = null;
    state.pages = [];
    $("#selected-content").classList.add("hidden");
    $("#selected-empty").classList.remove("hidden");
    $("#selected-badge").textContent = "No selection";
    $("#ask-scope").textContent = "All indexed documents";
    $("#ask-scope-hint").textContent = "Select a document in Overview to limit the answer to one file.";
    $("#tables-scope").textContent = "No document selected";
    await loadDocuments();
    setStatus("#upload-status", "Document deleted.", "success");
  } catch (error) {
    setStatus("#document-action-status", error.message, "error");
    button.disabled = false;
  }
}

async function uploadDocument() {
  const file = $("#pdf-file").files[0];
  if (!file) {
    setStatus("#upload-status", "Choose a PDF file first.", "error");
    return;
  }
  const button = $("#upload-button");
  button.disabled = true;
  button.textContent = "Uploading…";
  setStatus("#upload-status", "Extracting text and analysing page layouts…");
  try {
    const data = new FormData();
    data.append("file", file);
    const result = await api("/documents/upload", {method:"POST",body:data});
    await loadDocuments();
    await selectDocument(result.document_id);
    $("#pdf-file").value = "";
    setStatus("#upload-status", `Uploaded ${result.filename} · ${result.page_count} pages · ${formatNumber(result.character_count)} characters.`, "success");
  } catch (error) {
    setStatus("#upload-status", error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Upload document";
  }
}

function bindEvents() {
  $$(".nav-item").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
  $("#upload-button").addEventListener("click", uploadDocument);
  $("#search-form").addEventListener("submit", runSearch);
  $("#ask-form").addEventListener("submit", runAsk);
  $("#index-button").addEventListener("click", indexSelectedDocument);
  $("#tables-button").addEventListener("click", () => { showView("tables"); loadTables(); });
  $("#refresh-tables").addEventListener("click", loadTables);
  $("#delete-button").addEventListener("click", deleteSelectedDocument);
  $("#clear-ask-scope").addEventListener("click", () => {
    state.selectedId = null;
    state.selected = null;
    $("#ask-scope").textContent = "All indexed documents";
    $("#ask-scope-hint").textContent = "Questions will search across every indexed document.";
  });
}

async function init() {
  bindEvents();
  try {
    await loadDocuments();
  } catch (error) {
    setStatus("#upload-status", `Could not load document library: ${error.message}`, "error");
  }
}
document.addEventListener("DOMContentLoaded", init);
