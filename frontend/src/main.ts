import "./style.css";
type Method = { id: string; label: string };
type Evidence = {
  id: string;
  law_name: string;
  section_num: string;
  text: string;
};
type Run = {
  method: string;
  evidence: Evidence[];
  retrieved_ids: string[];
  answer: string;
  prompt_ids: string[];
  omitted_ids: string[];
  cited_ids: string[];
  invalid_ids: string[];
  retrieval_ms: number | null;
  generation_ms: number | null;
  ttft_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  error?: string;
  done: boolean;
};
type Entry = { kind: string; question: string; at: string; results: Run[] };
const $ = (s: string) => document.querySelector(s) as HTMLElement;
const esc = (s: unknown) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
let status = {
    ready: false,
    methods: [] as Method[],
    generator: null as string | null,
    corpus_size: 0,
    corpus_revision: null as string | null,
  },
  apiUnavailable = false,
  statusChecked = false,
  selected = "r1",
  controller: AbortController | null = null,
  current: Run[] = [],
  history: Entry[] = [];
const nav = [
  ["playground", "◈", "Playground"],
  ["compare", "▦", "Compare"],
  ["evaluate", "◎", "Evaluate"],
  ["history", "◷", "History"],
];
const route = () => {
  const requested = location.hash.slice(1);
  return nav.some(([id]) => id === requested) ? requested : "playground";
};
function shell() {
  let page = route();
  document.title = `${page[0].toUpperCase() + page.slice(1)} · Legal RAG`;
  $("#app").innerHTML =
    `<div class="app"><aside class="sidebar"><a href="#playground" class="brand"><span class="brandmark">§</span><span>Legal RAG<small>RESEARCH WORKSPACE</small></span></a><div class="navlabel">WORKSPACE</div><nav>${nav.map(([id, i, l]) => `<a class="navlink ${page === id ? "active" : ""}" href="#${id}" ${page === id ? 'aria-current="page"' : ""}><span class="navicon">${i}</span>${l}${id === "history" && history.length ? `<span class="navcount">${history.length}</span>` : ""}</a>`).join("")}</nav><div class="sidebar-foot"><span class="pulse ${status.ready ? "ready" : ""}"></span>${status.ready ? "เชื่อมต่อ API แล้ว" : apiUnavailable ? "API ยังไม่พร้อมใช้งาน" : statusChecked ? "ดัชนียังไม่พร้อม" : "กำลังตรวจสอบ API"}<small>THAI LEGAL CORPUS</small></div></aside><main class="main"><div class="topbar"><span>Thai legal research workspace</span><span class="api-status"><i class="pulse ${status.ready ? "ready" : ""}"></i>${status.ready ? "Backend พร้อมใช้งาน" : apiUnavailable ? "Backend ยังไม่พร้อมใช้งาน" : statusChecked ? "ดัชนียังไม่พร้อม" : "กำลังเชื่อมต่อ"}</span></div><div class="content">${page === "playground" ? playground() : page === "compare" ? compare() : page === "evaluate" ? evaluate() : historyPage()}</div></main></div>`;
  wire();
}
function head(k: string, t: string, d: string) {
  return `<header class="pagehead"><div class="eyebrow">${k}</div><h1>${t}</h1><p>${d}</p></header>`;
}
function form(compare = false) {
  return `<section class="card formcard"><div class="card-title"><div><h2>${compare ? "คำถามที่ต้องการเปรียบเทียบ" : "คำถามของคุณ"}</h2></div></div><label for="question">คำถามภาษาไทย</label><textarea id="question" rows="4" maxlength="1500" placeholder="เช่น ผู้เยาว์ทำสัญญาโดยไม่ได้รับความยินยอมจากผู้แทนโดยชอบธรรม มีผลอย่างไร?" aria-describedby="privacy count"></textarea><div class="inputmeta"><span id="privacy">โปรดหลีกเลี่ยงข้อมูลส่วนบุคคลที่ไม่จำเป็น</span><span id="count">0 / 1,500</span></div>${compare ? `<label class="label-gap">เลือก Retriever</label><div class="method-grid">${[1, 2, 3, 4].map((n) => `<label class="method-option"><input type="checkbox" name="methods" value="r${n}" ${status.methods.length && !status.methods.some((m) => m.id === `r${n}`) ? "disabled" : "checked"}><span>R${n}<small>${["Binary BoW", "TF-IDF", "Word2Vec", "BGE-M3"][n - 1]}</small></span></label>`).join("")}</div><label class="toggle"><input id="compare-generate" type="checkbox" ${status.generator ? "" : "disabled"}><span>สร้างคำตอบด้วย Generator</span><small>${status.generator ? "ปิดไว้เพื่อเปรียบเทียบความเร็วค้นคืน" : "ไม่มี Generator ใน backend นี้"}</small></label>` : `<label class="label-gap" for="retriever">Retriever</label><select id="retriever">${(status.methods.length ? status.methods : [1, 2, 3, 4].map((n) => ({ id: `r${n}`, label: `R${n} · ${["Binary Bag-of-Words", "TF-IDF", "Word2Vec", "BGE-M3"][n - 1]}` }))).map((m) => `<option value="${esc(m.id)}" ${m.id === selected ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select>`}<div class="actions"><button class="btn-primary" id="run" ${status.ready ? "" : "disabled"}>${compare ? "เริ่มเปรียบเทียบ" : status.generator ? "ค้นหลักฐานและสร้างคำตอบ" : "ค้นหลักฐาน"} <span>→</span></button><button class="btn-quiet" id="clear">ล้าง</button>${controller ? '<button class="btn-danger" id="cancel">ยกเลิก</button>' : ""}</div><div id="notice" role="status" aria-live="polite"></div></section>`;
}
function playground() {
  return `${head("WORKSPACE / PLAYGROUND", "ถามคำถามกฎหมาย", "พิมพ์คำถาม เลือก Retriever แล้วตรวจคำตอบพร้อมหลักฐานที่ค้นได้")}<div class="layout"><div>${form()}<div id="results">${runCard(current[0])}</div></div><aside class="sidecolumn"><div class="card info-card"><div class="minihead">SYSTEM STATUS</div><div class="systemrow"><span>Generator</span><b>${esc(status.generator || "ไม่พร้อมใช้งาน")}</b></div><div class="systemrow"><span>Corpus</span><b>${status.corpus_size ? status.corpus_size.toLocaleString() + " รายการ" : "—"}</b></div><div class="systemrow"><span>Revision</span><b>${esc(status.corpus_revision || "—")}</b></div></div><div class="notice-card"><b>ขอบเขตการใช้งาน</b><p>ผลลัพธ์เป็นข้อมูลจากการทดลอง ควรตรวจสอบตัวบทฉบับปัจจุบันและแหล่งทางการก่อนนำไปใช้</p></div></aside></div>`;
}
function compare() {
  return `${head("WORKSPACE / COMPARE", "เปรียบเทียบ Retriever", "ส่งคำถามเดียวกันให้หลายวิธี พร้อมเงื่อนไขการค้นคืนเดียวกัน")}${form(true)}<section class="compare-section"><div class="section-head"><div><div class="eyebrow">02 / RESULTS</div><h2>ผลการเปรียบเทียบ</h2></div><span class="muted">${current.length ? `${current.length} วิธี` : "ยังไม่มีผล"}</span></div><div class="compare-grid">${[
    1, 2, 3, 4,
  ]
    .map((n) =>
      runCard(
        current.find((r) => r.method === `r${n}`),
        true,
        n,
      ),
    )
    .join(
      "",
    )}</div><section class="card tablecard"><h2>สรุปหลักฐานและเวลา</h2><div class="table-wrap"><table><thead><tr><th>Retriever</th><th>Top-5</th><th>เข้า prompt</th><th>ถูกตัด</th><th>ค้นคืน</th><th>สร้างคำตอบ</th></tr></thead><tbody>${[
    1, 2, 3, 4,
  ]
    .map((n) => {
      const r = current.find((x) => x.method === `r${n}`);
      return `<tr><td>R${n}</td><td>${r ? esc(r.retrieved_ids.slice(0, 5).join(", ") || "—") : "—"}</td><td>${r ? esc(r.prompt_ids.join(", ") || "—") : "—"}</td><td>${r ? esc(r.omitted_ids.join(", ") || "—") : "—"}</td><td>${r?.retrieval_ms != null ? `${r.retrieval_ms.toFixed(0)} ms` : "—"}</td><td>${r?.generation_ms != null ? `${r.generation_ms.toFixed(0)} ms` : "—"}</td></tr>`;
    })
    .join(
      "",
    )}</tbody></table></div><p class="footnote">คำถามใหม่ไม่มี reference อัตโนมัติ จึงไม่คำนวณ Hit@5 หรือ MRR</p></section></section>`;
}
function evaluate() {
  return `${head("WORKSPACE / EVALUATE", "ประเมินผลการทดลอง", "ดูสรุป benchmark ที่ระบบตั้งค่าไว้และผลราย Retriever")}<section class="card eval-card"><div class="card-title"><div><h2>Benchmark summary</h2></div><span class="quiet-chip" id="eval-mode">กำลังโหลด</span></div><div id="eval-content" class="empty-state"><span class="empty-icon">◎</span><b>กำลังอ่านผลประเมิน</b></div></section><div class="notice-card eval-note"><b>แนวทางอ่านผล</b><p>การมี citation หรือพบ reference ไม่ได้ยืนยันว่าตัวบทสนับสนุนข้อสรุป ควรอ่านผล human review ควบคู่กัน</p></div>`;
}
function historyPage() {
  return `${head("WORKSPACE / HISTORY", "ประวัติการใช้งาน", "รายการ inference และ comparison ที่เก็บไว้ใน session นี้เท่านั้น")}<section class="card history-card"><div class="section-head"><div><h2>รายการล่าสุด</h2><p class="muted">ประวัติจะหายเมื่อปิดหรือรีเฟรชหน้านี้</p></div><button class="btn-quiet" id="clear-history" ${history.length ? "" : "disabled"}>ล้างประวัติ</button></div>${history.length ? `<div class="history-list">${history.map((e) => `<article class="history-item"><div class="history-mark">${e.kind === "Compare" ? "▦" : "◈"}</div><div class="history-body"><b>${esc(e.question)}</b><small>${e.kind} · ${new Date(e.at).toLocaleString("th-TH")} · ${e.results.length} วิธี</small></div><span class="history-ok">เสร็จสิ้น</span></article>`).join("")}</div>` : `<div class="empty-state"><span class="empty-icon">◷</span><b>ยังไม่มีประวัติ</b><p>ผลการทำงานจะปรากฏในหน้านี้ระหว่าง session ปัจจุบัน</p><a class="textlink" href="#playground">เริ่มค้นคำถาม →</a></div>`}</section>`;
}
function answerHtml(run: Run): string {
  const validCitations = new Set(
    run.cited_ids.filter((id) => run.prompt_ids.includes(id)),
  );
  return esc(run.answer)
    .replace(/\[(S\d{5})\]/g, (citation, id: string) =>
      validCitations.has(id)
        ? `<button class="cite-link" type="button" data-target="${id}">${citation}</button>`
        : citation,
    )
    .replace(/\n/g, "<br>");
}
function runCard(r?: Run, compact = false, n?: number) {
  if (!r)
    return `<section class="card answer-card ${compact ? "compact" : ""}"><div class="card-title"><div>${compact ? `<span class="step">R${n}</span>` : ""}<h2>${compact ? "ผลลัพธ์ Retriever" : "คำตอบ"}</h2></div><span class="quiet-chip">รอผล</span></div><div class="empty-state small-empty"><span class="empty-icon">✳</span><b>ผลลัพธ์จะแสดงที่นี่</b><p>หลังส่งคำถาม ระบบจะแสดงคำตอบและหลักฐานที่ค้นได้</p></div></section>`;
  return `<section class="card answer-card ${compact ? "compact" : ""}"><div class="card-title"><div>${compact ? `<span class="step">${esc(r.method.toUpperCase())}</span>` : ""}<h2>${compact ? "ผลลัพธ์" : "คำตอบ"}</h2></div><span class="quiet-chip ${r.error ? "error-chip" : ""}">${r.error ? "เกิดข้อผิดพลาด" : r.done ? "เสร็จสิ้น" : "กำลังประมวลผล"}</span></div>${r.error ? `<div class="error-box">${esc(r.error)}</div>` : `<div class="answer-text">${r.answer ? answerHtml(r) : r.done ? "ค้นคืนเสร็จแล้ว" : r.evidence.length ? "กำลังสร้างคำตอบ…" : "กำลังค้นหลักฐาน…"}</div>`}<div class="answer-meta"><span>${esc(r.method.toUpperCase())}</span><span>${r.retrieval_ms != null ? `ค้นคืน ${r.retrieval_ms.toFixed(0)} ms` : ""}</span>${r.generation_ms != null ? `<span>สร้าง ${r.generation_ms.toFixed(0)} ms</span>` : ""}</div><details class="evidence-details"><summary>หลักฐานที่ค้นได้ <span>${r.evidence.length} รายการ</span></summary><div class="evidence-list">${r.evidence.length ? r.evidence.map((e) => `<article class="evidence" data-evid="${esc(e.id)}"><div><b>${esc(e.law_name)}</b><span class="evidence-id">มาตรา ${esc(e.section_num)} · ${esc(e.id)}</span></div><p>${esc(e.text)}</p></article>`).join("") : '<p class="muted">ไม่มีหลักฐานที่ส่งกลับ</p>'}</div></details><details class="evidence-details trace"><summary>Trace และการใช้ token</summary><div class="tracegrid"><span>Top IDs</span><b>${esc(r.retrieved_ids.join(", ") || "—")}</b><span>เข้า prompt</span><b>${esc(r.prompt_ids.join(", ") || "—")}</b><span>ถูกตัดออก</span><b>${esc(r.omitted_ids.join(", ") || "—")}</b><span>Citations</span><b>${esc(r.cited_ids.join(", ") || "—")}</b><span>ไม่ถูกต้อง</span><b>${esc(r.invalid_ids.join(", ") || "—")}</b><span>TTFT / Tokens</span><b>${r.ttft_ms ?? "—"} ms · ${r.input_tokens ?? "—"} in / ${r.output_tokens ?? "—"} out</b></div></details></section>`;
}
function wire() {
  const q = document.querySelector("#question") as HTMLTextAreaElement | null;
  if (q) {
    q.addEventListener("input", () => {
      const c = $("#count");
      if (c) c.textContent = `${q.value.length} / 1,500`;
    });
  }
  const clear = document.querySelector("#clear");
  clear?.addEventListener("click", () => {
    if (q) q.value = "";
    current = [];
    shell();
  });
  document
    .querySelector("#retriever")
    ?.addEventListener(
      "change",
      (e) => (selected = (e.target as HTMLSelectElement).value),
    );
  document
    .querySelector("#run")
    ?.addEventListener("click", () => submit(route() === "compare"));
  document
    .querySelector("#cancel")
    ?.addEventListener("click", () => controller?.abort());
  document.querySelector("#clear-history")?.addEventListener("click", () => {
    history = [];
    shell();
  });
  if (route() === "evaluate") loadEvaluation();
}
function emptyRun(method: string): Run {
  return {
    method,
    evidence: [],
    retrieved_ids: [],
    answer: "",
    prompt_ids: [],
    omitted_ids: [],
    cited_ids: [],
    invalid_ids: [],
    retrieval_ms: null,
    generation_ms: null,
    ttft_ms: null,
    input_tokens: null,
    output_tokens: null,
    done: false,
  };
}
function updateResults() {
  if (route() === "compare") {
    const section = document.querySelector(".compare-section");
    if (section) {
      const fresh = compare();
      const tmp = document.createElement("div");
      tmp.innerHTML = fresh;
      const replacement = tmp.querySelector(".compare-section");
      if (replacement) section.replaceWith(replacement);
    }
  } else if (route() === "playground") {
    const out = document.querySelector("#results");
    if (out) out.innerHTML = runCard(current[0]);
  }
}
async function submit(isCompare: boolean) {
  const q = document.querySelector("#question") as HTMLTextAreaElement | null;
  const question = q?.value.trim() || "";
  const note = document.querySelector("#notice");
  if (!question) {
    if (note)
      note.innerHTML =
        '<span class="inline-error">กรุณาพิมพ์คำถามก่อนเริ่มค้นหา</span>';
    q?.focus();
    return;
  }
  const methods = isCompare
    ? Array.from(
        document.querySelectorAll<HTMLInputElement>(
          'input[name="methods"]:checked',
        ),
      ).map((x) => x.value)
    : [selected];
  if (!methods.length) {
    if (note)
      note.innerHTML =
        '<span class="inline-error">เลือก Retriever อย่างน้อยหนึ่งวิธี</span>';
    return;
  }
  if (!status.ready || controller) return;
  controller = new AbortController();
  current = methods.map(emptyRun);
  updateResults();
  if (note) note.textContent = "กำลังส่งคำถาม…";
  const generate = isCompare
    ? !!(document.querySelector("#compare-generate") as HTMLInputElement)
        ?.checked
    : !!status.generator;
  const questionEl = document.querySelector(
    "#question",
  ) as HTMLTextAreaElement | null;
  if (questionEl) questionEl.value = question;
  const runButton = document.querySelector("#run") as HTMLButtonElement | null;
  if (runButton) runButton.disabled = true;
  const actions = document.querySelector(".actions");
  if (actions)
    actions.insertAdjacentHTML(
      "beforeend",
      '<button class="btn-danger" id="cancel">ยกเลิก</button>',
    );
  document
    .querySelector("#cancel")
    ?.addEventListener("click", () => controller?.abort());
  if (note) note.textContent = "กำลังส่งคำถาม…";
  const thisController = controller;
  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify({ question, methods, generate }),
      signal: thisController.signal,
    });
    if (!res.ok)
      throw new Error(`เซิร์ฟเวอร์ตอบกลับ ${res.status} ${res.statusText}`);
    if (!res.body) throw new Error("ไม่พบข้อมูล streaming จากเซิร์ฟเวอร์");
    const reader = res.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    let streamComplete = false;
    function readEvent(line: string) {
      if (!line.trim() || controller !== thisController) return;
      const event = JSON.parse(line);
      if (event.type === "done") streamComplete = true;
      handleEvent(event);
    }
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) readEvent(line);
      if (done) break;
    }
    readEvent(buffer);
    if (controller !== thisController) return;
    if (!streamComplete) throw new Error("การเชื่อมต่อสิ้นสุดก่อนรับผลครบ");
    if (
      controller === thisController &&
      current.every((r) => r.done && !r.error)
    )
      history.unshift({
        kind: isCompare ? "Compare" : "Playground",
        question,
        at: new Date().toISOString(),
        results: current.map((r) => ({ ...r })),
      });
    if (note)
      note.textContent = current.some((r) => r.error)
        ? "มีผลลัพธ์ที่ประมวลผลไม่สำเร็จ"
        : "ประมวลผลเสร็จแล้ว";
  } catch (err) {
    if (controller !== thisController) return;
    if ((err as Error).name === "AbortError") {
      if (note) note.textContent = "ยกเลิกคำขอแล้ว";
    } else {
      const message =
        (err as Error).message || "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ";
      current.forEach((r) => {
        if (!r.done) {
          r.error = message;
          r.done = true;
        }
      });
      updateResults();
      if (note) note.textContent = `เกิดข้อผิดพลาด: ${message}`;
    }
  } finally {
    if (controller === thisController) {
      controller = null;
      const button = document.querySelector("#run") as HTMLButtonElement | null;
      if (button) button.disabled = false;
      document.querySelector("#cancel")?.remove();
    }
  }
}
function handleEvent(e: any) {
  if (e.type === "done") {
    return;
  }
  const r = current.find((x) => x.method === e.method);
  if (e.type === "error") {
    if (r) {
      r.error = String(e.message ?? "เกิดข้อผิดพลาด");
      r.done = true;
    } else
      current.forEach((x) => {
        x.error = String(e.message ?? "เกิดข้อผิดพลาด");
        x.done = true;
      });
  } else if (r && e.type === "retrieval") {
    r.evidence = Array.isArray(e.evidence) ? e.evidence : [];
    r.retrieved_ids = Array.isArray(e.retrieved_ids) ? e.retrieved_ids : [];
    r.retrieval_ms = Number.isFinite(e.retrieval_ms) ? e.retrieval_ms : null;
  } else if (r && e.type === "delta") {
    r.answer += String(e.text ?? "");
  } else if (r && e.type === "result") {
    r.answer = e.answer == null ? r.answer : String(e.answer);
    for (const k of [
      "prompt_ids",
      "omitted_ids",
      "cited_ids",
      "invalid_ids",
    ] as const)
      r[k] = Array.isArray(e[k]) ? e[k] : [];
    r.input_tokens = finite(e.input_tokens);
    r.output_tokens = finite(e.output_tokens);
    r.generation_ms = finite(e.generation_ms);
    r.ttft_ms = finite(e.ttft_ms);
    r.done = true;
  }
  updateResults();
}
function finite(n: unknown): number | null {
  return typeof n === "number" && Number.isFinite(n) ? n : null;
}
async function loadEvaluation() {
  const el = document.querySelector("#eval-content"),
    mode = document.querySelector("#eval-mode");
  try {
    const res = await fetch("/api/evaluate");
    if (!res.ok) throw new Error(`เซิร์ฟเวอร์ตอบกลับ ${res.status}`);
    const data = (await res.json()) as {
      mode?: unknown;
      rows?: Array<Record<string, unknown>>;
    };
    if (mode) mode.textContent = data.mode ? String(data.mode) : "SUMMARY";
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!el) return;
    if (!rows.length) {
      el.innerHTML =
        '<span class="empty-icon">◎</span><b>ยังไม่มีข้อมูลประเมิน</b><p>ไม่พบ retrieval summary ที่กำหนดไว้ใน backend</p>';
      return;
    }
    const keys = Array.from(
      new Set(rows.flatMap((r: Record<string, unknown>) => Object.keys(r))),
    );
    el.className = "eval-results";
    el.innerHTML = `<div class="eval-table-wrap"><table><thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${rows.map((r: Record<string, unknown>) => `<tr>${keys.map((k) => `<td>${esc(r[k] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p class="footnote">ข้อมูลจาก evaluation summary ที่ backend ตั้งค่าไว้${data.mode ? ` · ${esc(data.mode)}` : ""}</p>`;
  } catch (err) {
    if (el)
      el.innerHTML = `<span class="empty-icon">!</span><b>โหลดผลประเมินไม่สำเร็จ</b><p>${esc((err as Error).message)}</p>`;
    if (mode) mode.textContent = "OFFLINE";
  }
}
let lastRoute = route();
document.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>(".cite-link");
  if (!b) return;
  const card = b.closest(".answer-card");
  const details = card?.querySelector(
    ".evidence-details",
  ) as HTMLDetailsElement | null;
  if (details) details.open = true;
  const target = Array.from(
    card?.querySelectorAll<HTMLElement>(".evidence") || [],
  ).find((el) => el.dataset.evid === b.dataset.target);
  target?.scrollIntoView({ behavior: "smooth", block: "center" });
});
window.addEventListener("hashchange", () => {
  if (route() !== lastRoute) {
    controller?.abort();
    controller = null;
    current = [];
    lastRoute = route();
  }
  shell();
});
shell();
fetch("/api/status")
  .then((r) => {
    if (!r.ok) throw new Error();
    return r.json();
  })
  .then((s) => {
    statusChecked = true;
    status = {
      ready: !!s.ready,
      methods: Array.isArray(s.methods) ? s.methods : [],
      generator: s.generator ?? null,
      corpus_size: Number(s.corpus_size) || 0,
      corpus_revision: s.corpus_revision ?? null,
    };
    selected = status.methods[0]?.id ?? "r1";
    shell();
  })
  .catch(() => {
    status.ready = false;
    statusChecked = true;
    apiUnavailable = true;
    shell();
  });
