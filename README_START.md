# Start here (iPad Pro + keyboard, Codespaces)

1. Create a GitHub repo (e.g. `fusion`) and upload every file in this folder to its root
   (Working Copy app or the GitHub web upload). Keep the folder layout: CLAUDE.md at the root, docs/ etc.
2. Open the repo in a GitHub Codespace (2-core). In Safari: Share → Add to Home Screen, then open it
   from there so ⌘-shortcuts reach VS Code. Codespaces settings: idle timeout 15–30 min.
3. In the Codespace terminal install Claude Code following https://code.claude.com/docs/en/setup
   (native installer or npm), then `claude --version` and `claude doctor`.
4. `cd` to the repo root and run `claude`. It reads CLAUDE.md automatically.
5. Paste the block under "Kickoff prompt" from PROMPT_kickoff.md as the first message.
6. Answer each checkpoint (CP0–CP5) with the templates at the end of PROMPT_kickoff.md.
7. Later sessions: paste the "Resume prompt". Stop the codespace when you leave.

Budget: 4–8 h of Codespaces time; the free personal quota is 120 core-hours/month (60 h on 2-core).
Offline on the iPad (Pyto/Carnets): `python fusion_numpy.py` — the numpy twin, no torch needed.
