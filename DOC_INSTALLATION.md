# Documentation Installation Guide

This guide explains how to build, preview, and deploy the MkDocs documentation for Climb Analyzer.

## Prerequisites

- Python 3.8+
- pip

## Install MkDocs

```bash
pip install mkdocs-material mkdocstrings[python]
```

## Local Preview

Preview the documentation locally before deploying:

```bash
# From the repository root
mkdocs serve
```

Open http://localhost:8000 in your browser. Changes to docs files will auto-reload.

## Documentation Structure

```
climb-analyzer/
├── mkdocs.yml                    # MkDocs configuration
├── docs/                         # Documentation source
│   ├── index.md                  # Home page
│   ├── getting-started/          # Installation & quickstart
│   ├── user-guide/               # CLI, scoring, output format
│   ├── features/                 # Checkpointing, cloud cache, GUI
│   ├── data/                     # Elevation system docs
│   ├── advanced/                 # Configuration, GitHub App
│   ├── contributing/             # How to contribute
│   └── reference/                # API reference
└── .github/workflows/docs.yml    # Auto-deploy workflow
```

## Deploying to GitHub Pages

### Option 1: Automatic (GitHub Actions)

The documentation automatically deploys when you push changes to `main` branch that affect:
- `docs/**`
- `mkdocs.yml`
- `.github/workflows/docs.yml`

**First-time setup:**

1. Push the docs to main branch
2. Go to repository **Settings** → **Pages**
3. Under "Source", select **Deploy from a branch**
4. Select branch: `gh-pages` and folder: `/ (root)`
5. Click **Save**

The docs will be available at: `https://stevehollx.github.io/climb-analyzer/`

### Option 2: Manual Deploy

Deploy manually from your local machine:

```bash
# From the repository root
mkdocs gh-deploy --force
```

This builds the docs and pushes to the `gh-pages` branch.

## Editing Documentation

### Adding a New Page

1. Create the markdown file in the appropriate `docs/` subdirectory
2. Add it to the `nav` section in `mkdocs.yml`:

```yaml
nav:
  - Section Name:
    - Page Title: section/page-name.md
```

### Markdown Features

The documentation supports:

- **Admonitions** (notes, warnings, tips):
  ```markdown
  !!! note "Title"
      Content here

  !!! warning
      Warning content
  ```

- **Code blocks with syntax highlighting**:
  ```markdown
  ```python
  def example():
      pass
  ```
  ```

- **Tabs**:
  ```markdown
  === "Tab 1"
      Content for tab 1

  === "Tab 2"
      Content for tab 2
  ```

- **Tables**, **links**, **images** - standard markdown

### Theme Customization

Edit `mkdocs.yml` to customize:

```yaml
theme:
  name: material
  palette:
    primary: deep orange    # Primary color
    accent: orange          # Accent color
  features:
    - navigation.tabs       # Top navigation tabs
    - navigation.sections   # Collapsible sections
    - search.highlight      # Highlight search terms
```

## Building Static Site

Generate static HTML without deploying:

```bash
mkdocs build
```

Output is in the `site/` directory. Useful for:
- Offline documentation
- Custom hosting
- CI/CD integration

## Troubleshooting

### "Module not found" errors

Install missing dependencies:

```bash
pip install mkdocs-material mkdocstrings[python]
```

### Build fails on GitHub Actions

Check the workflow logs in the **Actions** tab. Common issues:
- Missing dependencies in workflow
- Invalid YAML in mkdocs.yml
- Broken links in markdown

### Pages not updating

1. Check GitHub Actions completed successfully
2. Wait a few minutes for GitHub Pages CDN
3. Hard refresh browser (Ctrl+Shift+R)
4. Check `gh-pages` branch has latest commit

### Local preview not working

```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
mkdocs serve -a localhost:8001
```

## Updating Dependencies

Periodically update MkDocs and plugins:

```bash
pip install --upgrade mkdocs-material mkdocstrings[python]
```

## Links

- **Live Docs**: https://stevehollx.github.io/climb-analyzer/
- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
