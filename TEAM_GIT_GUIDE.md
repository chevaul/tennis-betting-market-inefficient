# Team Git Guide
A quick reference for keeping our repo clean, synced, and conflict-free.

---

## 1. Always Pull Before You Push
**Command:**
```bash
git pull origin main
```
**Why:**  
Before committing or pushing, always pull the latest changes from GitHub.  
This ensures your local repo is up to date and prevents overwriting teammates’ work.

Tip: Think of it as syncing your notebook before editing.

---

## 2. Use Feature Branches — Don’t Code Directly on `main`
**Commands:**
```bash
# Create a new branch for your task
git checkout -b feature/eda-notebook

# After making changes
git add .
git commit -m "Add exploratory analysis for player stats"

# Push your branch to GitHub
git push origin feature/eda-notebook
```
**Why:**  
Each teammate works in their own branch for specific tasks (feature engineering, modeling, cleaning, etc.).  
Once complete, open a Pull Request (PR) to merge your branch into `main`.

Tip: `main` should always stay stable and runnable — no experiments directly there.

---

## 3. Commit Often, But Keep It Clean
**Commands:**
```bash
git add .
git commit -m "Clean data: handle missing odds and convert ranks to int"
git push origin feature/data-cleaning
```
**Why:**  
Frequent, descriptive commits make it easy to track changes and roll back mistakes.  
Each commit should summarize what and why, not how.

Tip: Avoid committing large raw data or temporary files (e.g., `.ipynb_checkpoints/`, `data/raw/`).

---

## Quick Checklist Before Every Push
- [ ] Pull latest `main`: `git pull origin main`
- [ ] Check status: `git status`
- [ ] Stage only intended changes: `git add <file>`
- [ ] Write a clear commit message
- [ ] Push to your feature branch: `git push origin feature/<task>`

---

## Extra Notes
- Never push your `data/raw/` folder — only processed or sample data.
- Use `.gitignore` to keep large or local-only files out of Git.
- Share dependencies by updating `requirements.txt` when you install a new library:
  ```bash
  pip freeze > requirements.txt
  ```

---

**Last updated:** November 2025  
**Maintainer:** Ecem Yildirim
