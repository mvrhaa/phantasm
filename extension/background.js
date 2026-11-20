// Phantasm's extension: connects to the brain (ws://127.0.0.1:<port>/browser) and runs what it asks
// in a Chrome window of Phantasm's own. The user's own tabs are never touched.

const PING_MS = 20000;
const LOAD_MS = 20000;
const MAX_TEXT = 20000;
const MAX_ELEMENTS = 300;

let socket = null;

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

async function setStatus(text) {
  await chrome.storage.local.set({ status: text });
  await chrome.action.setBadgeText({ text: text === "connected" ? "" : "!" });
}

async function connect() {
  if (socket && socket.readyState <= WebSocket.OPEN) return;
  const { port, token } = await chrome.storage.local.get({ port: 47400, token: "" });
  if (!token) {
    await setStatus("no token: paste the output of `phantasm token`");
    return;
  }
  const ws = new WebSocket(`ws://127.0.0.1:${port}/browser?token=${encodeURIComponent(token)}`);
  socket = ws;
  ws.onopen = () => setStatus("connected");
  ws.onclose = (event) => {
    if (socket === ws) socket = null;
    setStatus(event.code === 4401 ? "wrong token" : "not connected (is Phantasm running?)");
  };
  ws.onerror = () => {};
  ws.onmessage = async (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!message.id) return;
    let reply;
    try {
      reply = { id: message.id, result: await run(message.command, message.args || {}) };
    } catch (error) {
      reply = { id: message.id, error: String((error && error.message) || error) };
    }
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(reply));
  };
}

setInterval(() => {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping" }));
  else connect();
}, PING_MS);
chrome.alarms.create("phantasm-connect", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener(connect);
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
chrome.storage.onChanged.addListener((changes) => {
  if (!changes.port && !changes.token) return;
  if (socket) socket.close();
  socket = null;
  connect();
});
connect();

// ---------- Phantasm's window and tabs ----------

async function phantasmWindow() {
  const { windowId } = await chrome.storage.session.get({ windowId: null });
  if (windowId !== null) {
    try {
      await chrome.windows.get(windowId);
      return windowId;
    } catch {}
  }
  const created = await chrome.windows.create({ focused: false, width: 1280, height: 900 });
  await chrome.storage.session.set({ windowId: created.id });
  return created.id;
}

async function ownTab(id) {
  const tab = await chrome.tabs.get(Number(id));
  const { windowId } = await chrome.storage.session.get({ windowId: null });
  if (tab.windowId !== windowId) throw new Error(`tab ${id} is not one of Phantasm's tabs`);
  return tab;
}

async function loaded(id) {
  const start = Date.now();
  await sleep(150);
  while (Date.now() - start < LOAD_MS) {
    const tab = await chrome.tabs.get(id);
    if (tab.status === "complete") break;
    await sleep(200);
  }
  await sleep(300);
}

async function describe(id) {
  const tab = await chrome.tabs.get(id);
  return { tab: tab.id, url: tab.url, title: tab.title };
}

async function inPage(id, func, args) {
  const [frame] = await chrome.scripting.executeScript({ target: { tabId: id }, func, args });
  const result = frame ? frame.result : null;
  if (result && result.error) throw new Error(result.error);
  return result;
}

// ---------- Commands ----------

const COMMANDS = {
  async open({ url, tab }) {
    if (!/^[a-z]+:/i.test(url)) url = `https://${url}`;
    let id;
    if (tab) {
      id = (await ownTab(tab)).id;
      await chrome.tabs.update(id, { url });
    } else {
      const windowId = await phantasmWindow();
      const blank = await chrome.tabs.query({ windowId, url: "chrome://newtab/" });
      const created = await chrome.tabs.create({ windowId, url, active: true });
      id = created.id;
      for (const extra of blank) chrome.tabs.remove(extra.id).catch(() => {});
    }
    await loaded(id);
    return describe(id);
  },

  async read({ tab }) {
    const id = (await ownTab(tab)).id;
    return { tab: id, ...(await inPage(id, readPage, [MAX_TEXT, MAX_ELEMENTS])) };
  },

  async click({ tab, ref }) {
    const id = (await ownTab(tab)).id;
    await inPage(id, clickElement, [ref]);
    await loaded(id);
    return describe(id);
  },

  async type({ tab, ref, text, submit }) {
    const id = (await ownTab(tab)).id;
    await inPage(id, typeInto, [ref, text, Boolean(submit)]);
    await loaded(id);
    return describe(id);
  },

  async scroll({ tab, direction }) {
    const id = (await ownTab(tab)).id;
    await inPage(id, scrollPage, [direction === "up" ? -1 : 1]);
    return describe(id);
  },

  async back({ tab }) {
    const id = (await ownTab(tab)).id;
    await chrome.tabs.goBack(id);
    await loaded(id);
    return describe(id);
  },

  async screenshot({ tab }) {
    const own = await ownTab(tab);
    await chrome.tabs.update(own.id, { active: true });
    await sleep(300);
    const url = await chrome.tabs.captureVisibleTab(own.windowId, { format: "jpeg", quality: 70 });
    return { mime: "image/jpeg", data: url.split(",")[1] };
  },

  async tabs() {
    const { windowId } = await chrome.storage.session.get({ windowId: null });
    if (windowId === null) return [];
    try {
      const tabs = await chrome.tabs.query({ windowId });
      return tabs.map((tab) => ({ tab: tab.id, url: tab.url, title: tab.title }));
    } catch {
      return [];
    }
  },

  async close({ tab }) {
    await chrome.tabs.remove((await ownTab(tab)).id);
    return { closed: true };
  },
};

async function run(command, args) {
  const handler = COMMANDS[command];
  if (!handler) throw new Error(`unknown command ${command}`);
  return handler(args);
}

// ---------- Run inside the page ----------

function readPage(maxText, maxElements) {
  const selector = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary",
    "[role=button]", "[role=link]", "[role=checkbox]", "[role=tab]", "[role=menuitem]",
    "[contenteditable=true]",
  ].join(",");
  document.querySelectorAll("[data-phantasm-ref]").forEach((el) => el.removeAttribute("data-phantasm-ref"));
  const elements = [];
  for (const el of document.querySelectorAll(selector)) {
    const box = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (box.width === 0 || box.height === 0 || style.visibility === "hidden") continue;
    const ref = elements.length + 1;
    el.setAttribute("data-phantasm-ref", String(ref));
    const raw = el.getAttribute("aria-label") || el.innerText || el.value || el.placeholder ||
      el.title || el.alt || el.name || "";
    const field = el.tagName === "INPUT" || el.tagName === "TEXTAREA";
    elements.push({
      ref,
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute("type") || el.getAttribute("role") || "",
      label: String(raw).trim().replace(/\s+/g, " ").slice(0, 80),
      href: el.tagName === "A" ? el.href : "",
      value: field && el.type !== "password" ? String(el.value || "").slice(0, 80) : "",
    });
    if (elements.length >= maxElements) break;
  }
  const text = (document.body ? document.body.innerText : "").replace(/\n{3,}/g, "\n\n");
  return { url: location.href, title: document.title, text: text.slice(0, maxText), elements };
}

function clickElement(ref) {
  const el = document.querySelector(`[data-phantasm-ref="${ref}"]`);
  if (!el) return { error: `no element [${ref}]: read the page again` };
  el.scrollIntoView({ block: "center" });
  el.click();
  return { ok: true };
}

function typeInto(ref, text, submit) {
  const el = document.querySelector(`[data-phantasm-ref="${ref}"]`);
  if (!el) return { error: `no element [${ref}]: read the page again` };
  el.scrollIntoView({ block: "center" });
  el.focus();
  if (el.isContentEditable) {
    el.textContent = text;
    el.dispatchEvent(new InputEvent("input", { bubbles: true }));
  } else {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype
      : el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(el, text);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }
  if (submit) {
    const options = { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true };
    el.dispatchEvent(new KeyboardEvent("keydown", options));
    el.dispatchEvent(new KeyboardEvent("keyup", options));
    if (el.form && el.form.requestSubmit) el.form.requestSubmit();
  }
  return { ok: true };
}

function scrollPage(sign) {
  window.scrollBy(0, sign * window.innerHeight * 0.8);
  return { ok: true };
}
