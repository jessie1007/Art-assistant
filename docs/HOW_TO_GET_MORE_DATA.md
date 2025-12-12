# How to Get More Data: Modification Guide

If you're still not getting enough data, here are the key places to modify in `download_met_images.py`:

## 🔍 Where to Modify for More Data

### 1. **Expand Search Queries** (Line 129)

**Current:**
```python
ap.add_argument("--queries", nargs="*", default=["oil", "oil painting", "oil on canvas", "oil on wood","stilllife","painting", "portrait","landscape"])
```

**Modify to get MORE results:**
```python
ap.add_argument("--queries", nargs="*", default=[
    "oil", "oil painting", "oil on canvas", "oil on wood",
    "stilllife", "still life", "still-life",
    "painting", "paintings",
    "portrait", "portraits",
    "landscape", "landscapes",
    "watercolor", "water color", "water-colour",
    "tempera", "fresco", "mural",
    "drawing", "sketch", "charcoal",
    "pastel", "gouache",
    "impressionism", "impressionist",
    "realism", "realist",
    "abstract", "abstraction",
    "renaissance", "baroque", "rococo",
    "romanticism", "romantic",
    "modern", "contemporary",
    # Add more art movements/styles
])
```

**Why:** More queries = more artwork IDs found

---

### 2. **Remove Medium Restriction in Search** (Line 78)

**Current:**
```python
def search_ids_for_query(q: str) -> List[int]:
    params = {"q": q, "hasImages": "true", "medium": "Paintings"}
```

**Modify to get ALL artworks (not just paintings):**
```python
def search_ids_for_query(q: str) -> List[int]:
    # Remove "medium": "Paintings" to get all types
    params = {"q": q, "hasImages": "true"}
    # Or make it optional:
    # params = {"q": q, "hasImages": "true"}
    # if args.medium_filter:
    #     params["medium"] = "Paintings"
```

**Why:** Currently only searches "Paintings" - removing this gets drawings, sculptures, etc.

---

### 3. **Make Public Domain Filter Optional** (Line 100)

**Current:**
```python
def fetch_object(obj_id: int) -> Optional[Dict]:
    # ...
    if not obj.get("isPublicDomain", False):
        return None  # Always filters out non-public domain
```

**Modify to allow copyrighted works:**
```python
def fetch_object(obj_id: int, require_public_domain: bool = True) -> Optional[Dict]:
    # ...
    if require_public_domain and not obj.get("isPublicDomain", False):
        return None
    return obj

# In main(), add argument:
ap.add_argument("--allow-copyrighted", action="store_true", 
                help="Include copyrighted artworks (not just public domain)")

# In loop:
obj = fetch_object(oid, require_public_domain=not args.allow_copyrighted)
```

**Why:** Public domain filter removes many artworks - making it optional gets more

**⚠️ Warning:** Only use if you have rights to use copyrighted images!

---

### 4. **Remove Oil Filter Completely** (Line 153)

**Current:**
```python
if args.filter_oil:
    med = (obj.get("medium") or "").lower().replace("-", " ")
    if "oil" not in med:
        time.sleep(args.delay); continue
```

**Already fixed with `--no-oil-filter`**, but you can make it default:
```python
ap.add_argument("--oil-only", dest="filter_oil", action="store_true", default=False,
                help="Only include oil paintings (default: all paintings)")
```

**Why:** Default to getting all paintings, not just oil

---

### 5. **Increase Limit or Remove It** (Line 125)

**Current:**
```python
ap.add_argument("--limit", type=int, default=None, help="cap candidate IDs after union")
```

**Modify to process more:**
```python
# Option 1: Increase default limit
ap.add_argument("--limit", type=int, default=50000, help="cap candidate IDs after union")

# Option 2: Add --no-limit flag
ap.add_argument("--no-limit", action="store_true", help="Process all found IDs (no limit)")
```

**In fetch_ids_union:**
```python
def fetch_ids_union(queries: Iterable[str], limit: Optional[int] = None, no_limit: bool = False) -> List[int]:
    seen = set()
    for q in queries:
        for oid in search_ids_for_query(q):
            seen.add(oid)
    ids = sorted(seen)
    if not no_limit and limit:
        ids = ids[:limit]
    return ids
```

**Why:** Process more artwork IDs

---

### 6. **Add More Search Strategies** (New Function)

**Add this to search more comprehensively:**
```python
def search_by_department(department: str) -> List[int]:
    """Search by museum department (e.g., 'European Paintings', 'American Paintings')"""
    params = {"departmentId": department, "hasImages": "true"}
    r = SESSION.get(MET_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("objectIDs") or []

def search_by_date_range(start_year: int, end_year: int) -> List[int]:
    """Search artworks from specific date range"""
    # Note: Met API doesn't directly support this, but you can filter after fetching
    # This would require fetching and filtering
    pass
```

**Why:** Different search strategies find different artworks

---

### 7. **Remove Image URL Requirement** (Line 159)

**Current:**
```python
url = obj.get("primaryImageSmall") if args.thumb else (obj.get("primaryImage") or obj.get("primaryImageSmall"))
if not url:
    time.sleep(args.delay); continue  # Skips if no image
```

**Modify to include artworks without images:**
```python
ap.add_argument("--include-no-image", action="store_true",
                help="Include artworks even if they don't have images")

# In loop:
url = obj.get("primaryImageSmall") if args.thumb else (obj.get("primaryImage") or obj.get("primaryImageSmall"))
if not url:
    if args.include_no_image:
        url = None  # Allow None
        local_path = None
    else:
        time.sleep(args.delay); continue
```

**Why:** Some artworks don't have images but have metadata

---

### 8. **Add Department/Culture Filters** (New)

**Add arguments to search specific departments:**
```python
ap.add_argument("--departments", nargs="*", 
                default=["European Paintings", "American Paintings", "Modern and Contemporary Art"],
                help="Museum departments to search")

# Then search each department
for dept in args.departments:
    dept_ids = search_by_department(dept)
    ids.extend(dept_ids)
```

**Why:** Different departments have different artworks

---

## 🎯 Recommended Modifications (Priority Order)

### **High Impact (Do These First):**

1. **Remove medium restriction** (Line 78)
   ```python
   # Change from:
   params = {"q": q, "hasImages": "true", "medium": "Paintings"}
   # To:
   params = {"q": q, "hasImages": "true"}
   ```

2. **Expand queries** (Line 129)
   - Add more art movements, styles, techniques

3. **Make public domain optional** (Line 100)
   - Add `--allow-copyrighted` flag

### **Medium Impact:**

4. **Increase/remove limit** (Line 125)
   - Set higher default or add `--no-limit`

5. **Add department searches** (New)
   - Search by museum department

### **Low Impact (But Helpful):**

6. **Allow no-image artworks** (Line 159)
   - Include metadata even without images

7. **Add more search terms** (Line 129)
   - More specific queries

---

## 📝 Complete Modified Version (Key Changes)

Here's a modified version with the most impactful changes:

```python
# Line 78: Remove medium restriction
def search_ids_for_query(q: str, medium_filter: bool = False) -> List[int]:
    params = {"q": q, "hasImages": "true"}
    if medium_filter:
        params["medium"] = "Paintings"
    r = SESSION.get(MET_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("objectIDs") or []

# Line 100: Make public domain optional
def fetch_object(obj_id: int, require_public_domain: bool = True) -> Optional[Dict]:
    try:
        r = SESSION.get(MET_OBJECT.format(objectID=obj_id), timeout=30)
        if r.status_code != 200:
            return None
        obj = r.json()
        if require_public_domain and not obj.get("isPublicDomain", False):
            return None
        return obj
    except Exception:
        return None

# Line 125: Add no-limit option
ap.add_argument("--no-limit", action="store_true", help="Process all IDs (no limit)")

# Line 129: Expanded queries
ap.add_argument("--queries", nargs="*", default=[
    "oil", "oil painting", "watercolor", "tempera", "fresco",
    "painting", "portrait", "landscape", "still life",
    "impressionism", "realism", "abstract", "renaissance",
    "baroque", "romanticism", "modern", "contemporary",
    # Add more...
])

# Line 131: Add allow copyrighted
ap.add_argument("--allow-copyrighted", action="store_true",
                help="Include copyrighted artworks")

# Line 132: Change default to no oil filter
ap.add_argument("--oil-only", dest="filter_oil", action="store_true", default=False,
                help="Only include oil paintings")

# In main() loop:
ids = fetch_ids_union(args.queries, limit=args.limit if not args.no_limit else None)
# ...
obj = fetch_object(oid, require_public_domain=not args.allow_copyrighted)
```

---

## 🚀 Expected Results After Modifications

| Modification | Expected Increase |
|--------------|-------------------|
| Remove medium restriction | +50-100% more IDs |
| Expand queries | +30-50% more IDs |
| Allow copyrighted | +200-300% more IDs (⚠️ legal issues) |
| Remove limit | Process all found IDs |
| Add departments | +20-40% more IDs |

**Combined:** Could go from 5851 IDs → 20,000-50,000+ IDs

---

## ⚠️ Important Considerations

1. **Copyright:** Including copyrighted works may have legal restrictions
2. **API Rate Limits:** More requests = longer runtime, possible rate limiting
3. **Storage:** More images = more disk space needed
4. **Processing Time:** More data = longer embedding/indexing time

---

## 🎯 Quick Win: Minimal Changes for Maximum Impact

**Just change these 2 lines:**

1. **Line 78:** Remove `"medium": "Paintings"` restriction
2. **Line 129:** Add more queries to default list

This alone should double or triple your results!

