# React Dashboard - All Files Ready to Download

**Created by Cazandra Aporbo | Data Science**

---

## Download All Files Below (Click Each Link)

### Configuration Files (Root Directory)

1. [package.json](computer:///mnt/user-data/outputs/dashboard_package.json) - Dependencies and scripts
2. [index.html](computer:///mnt/user-data/outputs/dashboard_index.html) - HTML entry point  
3. [vite.config.js](computer:///mnt/user-data/outputs/dashboard_vite.config.js) - Build configuration
4. [README.md](computer:///mnt/user-data/outputs/DASHBOARD_README.md) - Complete documentation

### Source Files (src/ directory)

5. [src/index.jsx](computer:///mnt/user-data/outputs/dashboard_src_index.jsx) - React entry point
6. [src/App.jsx](computer:///mnt/user-data/outputs/dashboard_src_App.jsx) - Main application
7. [src/theme.js](computer:///mnt/user-data/outputs/dashboard_src_theme.js) - Color palette & design

### Styles (src/styles/ directory)

8. [src/styles/global.css](computer:///mnt/user-data/outputs/dashboard_global.css) - Global CSS

### Components (src/components/ directory)

9. [src/components/Header.jsx](computer:///mnt/user-data/outputs/dashboard_Header.jsx)
10. [src/components/Navigation.jsx](computer:///mnt/user-data/outputs/dashboard_Navigation.jsx)
11. [src/components/Dashboard.jsx](computer:///mnt/user-data/outputs/dashboard_Dashboard.jsx)
12. [src/components/Footer.jsx](computer:///mnt/user-data/outputs/dashboard_Footer.jsx)

### Section Components (src/components/sections/ directory)

13. [src/components/sections/Overview.jsx](computer:///mnt/user-data/outputs/dashboard_Overview.jsx)
14. [src/components/sections/TemporalAnalysis.jsx](computer:///mnt/user-data/outputs/dashboard_TemporalAnalysis.jsx)
15. [src/components/sections/GenderAnalysis.jsx](computer:///mnt/user-data/outputs/dashboard_GenderAnalysis.jsx)
16. [src/components/sections/GeospatialAnalysis.jsx](computer:///mnt/user-data/outputs/dashboard_GeospatialAnalysis.jsx)

### Additional Sections (export from GeospatialAnalysis.jsx)

The following import from GeospatialAnalysis.jsx:
- CollaborationAnalysis
- NLPAnalysis  
- MLAnalysis
- StatisticalAnalysis

---

## Setup Instructions

### Step 1: Create Project Structure

```bash
mkdir nobel-prize-dashboard
cd nobel-prize-dashboard

# Create directories
mkdir -p src/components/sections
mkdir -p src/styles
```

### Step 2: Download and Place Files

Click each link above and save to the correct directory:

**Root directory files:**
- package.json
- index.html
- vite.config.js
- README.md

**src/ directory:**
- index.jsx
- App.jsx
- theme.js

**src/styles/ directory:**
- global.css

**src/components/ directory:**
- Header.jsx
- Navigation.jsx
- Dashboard.jsx
- Footer.jsx

**src/components/sections/ directory:**
- Overview.jsx
- TemporalAnalysis.jsx
- GenderAnalysis.jsx
- GeospatialAnalysis.jsx

### Step 3: Create Export Files

```bash
# In src/components/sections/ create these files:

# CollaborationAnalysis.jsx
echo "export { CollaborationAnalysis as default } from './GeospatialAnalysis'" > src/components/sections/CollaborationAnalysis.jsx

# NLPAnalysis.jsx
echo "export { NLPAnalysis as default } from './GeospatialAnalysis'" > src/components/sections/NLPAnalysis.jsx

# MLAnalysis.jsx
echo "export { MLAnalysis as default } from './GeospatialAnalysis'" > src/components/sections/MLAnalysis.jsx

# StatisticalAnalysis.jsx
echo "export { StatisticalAnalysis as default } from './GeospatialAnalysis'" > src/components/sections/StatisticalAnalysis.jsx
```

### Step 4: Copy Dataset

```bash
# Copy your CSV file to project root
cp /path/to/nobel_prizes_1901-2025_cleaned.csv .
```

### Step 5: Install Dependencies

```bash
npm install
```

This installs:
- react 18.2.0
- react-dom 18.2.0
- recharts 2.10.0
- papaparse 5.4.1
- styled-components 6.1.0
- vite 5.0.0
- @vitejs/plugin-react 4.2.0

### Step 6: Start Development Server

```bash
npm run dev
```

Dashboard opens at: **http://localhost:3000**

### Step 7: Build for Production

```bash
npm run build
```

Production files in: **dist/**

---

## File Structure After Setup

```
nobel-prize-dashboard/
├── package.json
├── index.html
├── vite.config.js
├── README.md
├── nobel_prizes_1901-2025_cleaned.csv
└── src/
    ├── index.jsx
    ├── App.jsx
    ├── theme.js
    ├── styles/
    │   └── global.css
    └── components/
        ├── Header.jsx
        ├── Navigation.jsx
        ├── Dashboard.jsx
        ├── Footer.jsx
        └── sections/
            ├── Overview.jsx
            ├── TemporalAnalysis.jsx
            ├── GenderAnalysis.jsx
            ├── GeospatialAnalysis.jsx
            ├── CollaborationAnalysis.jsx
            ├── NLPAnalysis.jsx
            ├── MLAnalysis.jsx
            └── StatisticalAnalysis.jsx
```

---

## Features

- 8 Interactive Sections
- Real-time CSV Data Loading
- Responsive Charts (Recharts)
- Your Custom Color Palette
- Professional Attribution
- Production-Ready

---

## Troubleshooting

**CSV Not Loading?**
- Ensure CSV is in project root
- Check filename matches in App.jsx

**Port Already in Use?**
- Change port in vite.config.js

**Dependencies Not Installing?**
```bash
rm -rf node_modules package-lock.json
npm install
```

---

## Author

**Cazandra Aporbo**  


MS in Data Science, University of Denver  
BS in Integrative Biology, Oregon State University

Specialization: AI Ethics, Bias Detection, Healthcare Analytics

---

**Created:** November 3, 2025  
**Version:** 1.0  
**Technology:** React 18 + Vite 5 + Recharts 2.10
