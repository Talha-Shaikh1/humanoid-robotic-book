# GitHub Pages Deployment Instructions

## Prerequisites
- Node.js 18+ installed
- Git installed
- GitHub account with repository access

## Setup Process

### 1. Install Dependencies
```bash
cd docusaurus
npm install
```

### 2. Build the Site
```bash
npm run build
```

### 3. Deploy to GitHub Pages
```bash
# Set environment variables
export GIT_USER=<Your GitHub Username>
export USE_SSH=false  # or true if using SSH

# Deploy the site
npm run deploy
```

## GitHub Actions Deployment (Recommended)

### 1. Create GitHub Actions Workflow
Create `.github/workflows/deploy.yml` with the following content:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm

      - name: Install dependencies
        run: npm ci
      - name: Build website
        run: npm run build

      # Popular action to deploy to GitHub Pages:
      # Docs: https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-docusaurus
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          # Build output to publish to the `gh-pages` branch:
          publish_dir: ./build
          # The following lines assign commit authorship to the official
          # GH-Actions bot for deploys to `gh-pages` branch:
          # https://github.com/actions/checkout/issues/13#issuecomment-724415212
          # The GH actions bot is used by default if you didn't specify these arguments
          user_name: github-actions[bot]
          user_email: 41898282+github-actions[bot]@users.noreply.github.com
```

### 2. Configure GitHub Repository
1. Go to your repository Settings
2. Navigate to Pages section
3. Select source as "Deploy from a branch"
4. Select branch `gh-pages` and folder `/ (root)`

## Manual Deployment Process

### 1. Clone the Repository
```bash
git clone https://github.com/<username>/<repository-name>.git
cd <repository-name>
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Build the Site
```bash
npm run build
```

### 4. Deploy
```bash
GIT_USER=<Your GitHub Username> USE_SSH=true npm run deploy
```

## Post-Deployment Validation

### 1. Verify Site is Live
- Check that the site is accessible at `https://<username>.github.io/<repository-name>/`
- Verify all pages load correctly

### 2. Test Functionality
- Navigate through all 4 modules
- Verify all internal links work
- Test search functionality
- Verify diagrams render correctly
- Check that code examples are displayed properly

### 3. Mobile Responsiveness
- Test site on different screen sizes
- Verify navigation works on mobile devices
- Check that diagrams scale appropriately

## Troubleshooting

### Build Issues
- Ensure all dependencies are installed: `npm install`
- Clear Docusaurus cache: `npm run clear`
- Check for syntax errors in markdown files

### Deployment Issues
- Verify GIT_USER environment variable is set
- Check GitHub repository permissions
- Ensure the gh-pages branch exists

### Content Issues
- Verify all image paths are correct
- Check that all internal links resolve
- Ensure all code examples are properly formatted