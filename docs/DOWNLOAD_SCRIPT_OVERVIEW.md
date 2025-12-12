# High-Level Overview: download_met_images.py

## What This Script Does (Big Picture)

**Goal:** Download artwork images from Met Museum API and save metadata to a parquet file.

**Input:** Search queries (e.g., "oil painting", "portrait")
**Output:** 
- Downloaded images in `data/images/`
- Metadata file: `data/corpus/met_oil_paintings.parquet`

---

## Section-by-Section Breakdown

### 📦 **Section 1: Imports & Setup** (Lines 1-35)

**What it does:**
- Imports libraries (requests, pandas, PIL, etc.)
- Sets up HTTP session with retry logic
- Defines constants (API URLs, regex patterns)

**Key components:**
```python
SESSION = make_session()  # HTTP client with auto-retry
YEAR_RE = re.compile(...)  # Pattern to extract years from dates
```

**Why it matters:**
- Provides tools needed for the rest of the script
- Handles network errors automatically

---

### 🔧 **Section 2: Helper Functions** (Lines 36-75)

**What it does:**
- Small utility functions used throughout the script

**Functions:**

1. **`make_session()`** - Creates HTTP client with retry logic
   - Handles network failures automatically
   - Waits and retries if server is busy

2. **`extract_year()`** - Extracts year from date strings
   - Input: "1776–1783" → Output: 1776
   - Used for filtering by year

3. **`to_century_bin()`** - Groups years into centuries
   - Input: 1776 → Output: "1700s"
   - Used for organizing data

4. **`ensure_dirs()`** - Creates necessary folders
   - Makes sure `data/images/` and `data/corpus/` exist

5. **`_safe_name()`** - Creates safe filenames
   - Converts URLs to hash-based filenames
   - Prevents invalid characters in filenames

6. **`expected_local_path()`** - Calculates where image should be saved
   - Input: image URL → Output: "data/images/abc123.jpg"

**Why it matters:**
- Reusable code (DRY principle)
- Makes main logic cleaner

---

### 🌐 **Section 3: API Interaction Functions** (Lines 77-123)

**What it does:**
- Functions that talk to Met Museum API

**Functions:**

1. **`search_ids_for_query(query)`** (Lines 77-83)
   - **Input:** Search term (e.g., "oil painting")
   - **Output:** List of artwork IDs
   - **How:** Calls Met API search endpoint
   - **Example:** "oil painting" → [474, 1234, 5678, ...]

2. **`fetch_ids_union(queries, limit)`** (Lines 85-93)
   - **Input:** Multiple search queries, optional limit
   - **Output:** Combined list of unique IDs
   - **How:** Runs multiple searches, combines results, removes duplicates
   - **Example:** ["oil", "portrait"] → [474, 1234, 5678, 9999, ...]

3. **`fetch_object(obj_id)`** (Lines 95-105)
   - **Input:** Artwork ID (e.g., 474)
   - **Output:** Full artwork data (JSON) or None
   - **How:** Calls Met API object endpoint
   - **Filters:** Only returns if `isPublicDomain = True`
   - **Example:** 474 → {title: "...", artist: "...", ...}

4. **`download_and_cache(url, longest_side)`** (Lines 107-122)
   - **Input:** Image URL, max size
   - **Output:** Local file path or None
   - **How:** 
     1. Checks if already downloaded (skip if exists)
     2. Downloads image from URL
     3. Resizes to max dimension
     4. Saves as JPEG
   - **Example:** "https://..." → "data/images/abc123.jpg"

**Why it matters:**
- Separates API logic from business logic
- Handles errors gracefully (returns None on failure)

---

### 🎯 **Section 4: Main Function - Setup** (Lines 124-142)

**What it does:**
- Parses command-line arguments
- Sets up initial data structures

**Key steps:**
```python
ap = argparse.ArgumentParser()  # Define command-line options
ap.add_argument("--limit", ...)  # How many artworks to process
ap.add_argument("--download-missing", ...)  # Download images?
ap.add_argument("--queries", ...)  # What to search for
# ... more arguments

args = ap.parse_args()  # Read user's command-line input
ensure_dirs()  # Create folders if needed
ids = fetch_ids_union(args.queries, limit=args.limit)  # Get artwork IDs
```

**Why it matters:**
- Makes script flexible (user controls behavior)
- Validates input before processing

---

### 🔄 **Section 5: Main Loop - Process Each Artwork** (Lines 144-204)

**What it does:**
- Loops through each artwork ID
- Filters, downloads, and collects metadata

**Flow for each artwork:**

```
For each artwork ID:
  ↓
1. Fetch artwork data from API
   ↓
2. Check if public domain (skip if not)
   ↓
3. Filter by medium (skip if not "oil" - unless --no-oil-filter)
   ↓
4. Get image URL (primaryImage or primaryImageSmall)
   ↓
5. Parse year and century
   ↓
6. Filter by year range (if --min-year/--max-year specified)
   ↓
7. Check if image exists locally
   ↓
8. Download image if needed (if --download-missing)
   ↓
9. Create metadata dictionary
   ↓
10. Add to rows list
```

**Key filtering steps:**
- **Line 146:** `fetch_object()` - Get data, skip if not public domain
- **Line 150-152:** Filter by medium (oil paintings)
- **Line 154:** Get image URL
- **Line 163-166:** Filter by year range
- **Line 171-184:** Handle image download

**Why it matters:**
- Core business logic
- Applies all filters and downloads images
- Builds the metadata collection

---

### 💾 **Section 6: Save Results** (Lines 206-237)

**What it does:**
- Converts rows to DataFrame
- Merges with existing data
- Normalizes data types
- Saves to parquet file

**Steps:**

1. **Convert to DataFrame** (Line 206)
   ```python
   new_df = pd.DataFrame(rows)  # List of dicts → DataFrame
   ```

2. **Merge with existing data** (Lines 208-214)
   ```python
   if existing file:
       old_df = read existing parquet
       merged = combine old + new
       remove duplicates
   ```

3. **Normalize data types** (Lines 214-232)
   - Convert numbers to proper types
   - Convert strings to string type
   - Reorder columns

4. **Save to parquet** (Line 236)
   ```python
   merged.to_parquet(PARQUET, index=False)
   ```

**Why it matters:**
- Preserves data from previous runs
- Ensures consistent data format
- Creates final output file

---

## How Sections Connect (Data Flow)

```
┌─────────────────────────────────────────────────────────┐
│ SECTION 1: Setup                                         │
│ - Import libraries                                       │
│ - Create HTTP session                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ SECTION 2: Helper Functions                              │
│ - Utility functions (extract_year, etc.)                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ SECTION 3: API Functions                                 │
│ - search_ids_for_query() → Get artwork IDs              │
│ - fetch_ids_union() → Combine multiple searches         │
│ - fetch_object() → Get artwork details                 │
│ - download_and_cache() → Download images                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ SECTION 4: Main Setup                                    │
│ - Parse command-line arguments                          │
│ - Get list of artwork IDs                               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ SECTION 5: Main Loop                                     │
│ FOR each artwork ID:                                    │
│   1. fetch_object() → Get data                          │
│   2. Filter (public domain, medium, year)               │
│   3. download_and_cache() → Download image              │
│   4. Create metadata dict → Add to rows[]               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ SECTION 6: Save Results                                  │
│ - Convert rows[] to DataFrame                           │
│ - Merge with existing data                              │
│ - Save to parquet file                                  │
└─────────────────────────────────────────────────────────┘
```

## Complete Example Flow

**User runs:**
```bash
python scripts/download_met_images.py --limit 100 --download-missing --queries "oil painting" "portrait"
```

**What happens:**

1. **Section 4:** Parse arguments → `limit=100`, `queries=["oil painting", "portrait"]`

2. **Section 3:** `fetch_ids_union()` → Search API for both queries → Get 1000+ IDs → Limit to 100

3. **Section 5 (Loop):** For each of 100 IDs:
   - `fetch_object(474)` → Get artwork data
   - Check: Is public domain? ✅
   - Check: Medium contains "oil"? ✅
   - Get image URL
   - `download_and_cache()` → Download image → Save to `data/images/abc123.jpg`
   - Create metadata dict → Add to `rows[]`

4. **Section 6:** 
   - Convert `rows[]` (100 dicts) → DataFrame
   - Merge with existing parquet (if exists)
   - Save → `data/corpus/met_oil_paintings.parquet`

**Result:**
- 100 images in `data/images/`
- Metadata for 100 artworks in parquet file

---

## Key Design Patterns

### 1. **Separation of Concerns**
- API functions (Section 3) separate from business logic (Section 5)
- Helper functions (Section 2) reusable across script

### 2. **Error Handling**
- Functions return `None` on failure (graceful degradation)
- Script continues even if some artworks fail

### 3. **Incremental Processing**
- Checks if image exists before downloading (avoids re-download)
- Merges with existing parquet (preserves previous runs)

### 4. **Flexibility**
- Command-line arguments control behavior
- Filters can be enabled/disabled

---

## Summary

| Section | Purpose | Key Output |
|---------|---------|------------|
| **1. Setup** | Import libraries, create session | HTTP client ready |
| **2. Helpers** | Utility functions | Reusable tools |
| **3. API** | Talk to Met Museum API | Get artwork data, download images |
| **4. Main Setup** | Parse arguments, get IDs | List of artwork IDs |
| **5. Main Loop** | Process each artwork | `rows[]` list of metadata |
| **6. Save** | Convert and save data | Parquet file |

**Overall Flow:**
```
Arguments → Get IDs → Process Each → Collect Metadata → Save to File
```

