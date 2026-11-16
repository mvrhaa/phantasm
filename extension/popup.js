// The extension's settings: Zero's port and token (the output of `zero token`).
const port = document.getElementById("port");
const token = document.getElementById("token");
const status = document.getElementById("status");

async function show() {
  const saved = await chrome.storage.local.get({ port: 47400, token: "", status: "not connected" });
  port.value = saved.port;
  token.value = saved.token;
  status.textContent = saved.status;
}

document.getElementById("save").addEventListener("click", async () => {
  await chrome.storage.local.set({ port: Number(port.value) || 47400, token: token.value.trim() });
  status.textContent = "saved, connecting…";
});

chrome.storage.onChanged.addListener(show);
show();
