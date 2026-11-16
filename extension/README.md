# Zero's Chrome extension

Lets Zero use your Chrome — signed in as you are — in a window of its own when you ask it to. Your own tabs are never touched.

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose this folder.
2. Run `zero token` (in the project: `uv run zero token`) and copy the token.
3. Click the extension's icon, paste the token (the port is 47400 unless you changed `server.port`) and press **Save**.

The popup says **connected** while Zero runs. Zero reads pages without asking; clicking and typing ask you first unless you allow them.
