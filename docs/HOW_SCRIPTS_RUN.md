# How Python Scripts Run: Understanding `if __name__ == "__main__"`

## The Question

You asked: "Where does it call to run this script? I thought you will call 'main' in order to execute all this script"

Great question! Let me explain how Python scripts execute.

---

## How Python Scripts Execute

### When You Run a Script Directly

When you run:
```bash
python scripts/download_met_images.py --limit 1000
```

Python does this:

1. **Reads the entire file** (top to bottom)
2. **Executes all code** at the module level (imports, function definitions, etc.)
3. **Checks the `if __name__ == "__main__":` block** at the bottom
4. **If True, runs the code inside** (usually calls `main()`)

### The Magic: `if __name__ == "__main__":`

Look at the bottom of `download_met_images.py`:

```python
# Line 247-248
if __name__ == "__main__":
    main()
```

**What this means:**
- `__name__` is a special Python variable
- When you run the script directly: `__name__ == "__main__"` → **True**
- When you import it: `__name__ == "download_met_images"` → **False**

**So:**
- ✅ Run directly → `main()` is called
- ❌ Import as module → `main()` is NOT called

---

## Complete Execution Flow

### Scenario 1: Running Script Directly

```bash
python scripts/download_met_images.py --limit 1000
```

**What happens:**

```
1. Python starts reading the file
   ↓
2. Lines 1-30: Execute imports
   import os, io, time, argparse, hashlib
   import requests
   import pandas as pd
   ...
   ↓
3. Lines 32-47: Execute module-level code
   HEADERS = {...}
   SESSION = make_session()
   ...
   ↓
4. Lines 49-121: Define functions (NOT executed yet)
   def extract_year(...):
       ...
   def fetch_object(...):
       ...
   def download_and_cache(...):
       ...
   ↓
5. Lines 123-245: Define main() function (NOT executed yet)
   def main():
       ap = argparse.ArgumentParser()
       ...
   ↓
6. Lines 247-248: Check if running directly
   if __name__ == "__main__":  # True!
       main()  # ← THIS IS WHERE main() GETS CALLED!
   ↓
7. main() executes:
   - Parse arguments
   - Get artwork IDs
   - Process each artwork
   - Save results
```

### Scenario 2: Importing as Module

```python
# In another Python file
from scripts.download_met_images import fetch_object

# Use the function
obj = fetch_object(474)
```

**What happens:**

```
1. Python reads the file
   ↓
2. Executes imports and module-level code
   ↓
3. Defines all functions (including main())
   ↓
4. Checks if __name__ == "__main__"
   # False! (__name__ == "download_met_images")
   ↓
5. Skips the if block
   # main() is NOT called
   ↓
6. You can now use functions like fetch_object()
```

---

## Why This Pattern Exists

### Problem Without `if __name__ == "__main__":`

```python
# Bad example (without the check)
def main():
    print("Running script...")
    # ... do stuff ...

main()  # Always runs, even when imported!
```

**If you import this:**
```python
from bad_script import some_function
# Oops! main() runs automatically, even though you just wanted to import!
```

### Solution With `if __name__ == "__main__":`

```python
# Good example (with the check)
def main():
    print("Running script...")
    # ... do stuff ...

if __name__ == "__main__":
    main()  # Only runs when script is executed directly
```

**Now:**
- ✅ Run directly → `main()` executes
- ✅ Import as module → `main()` doesn't execute, but functions are available

---

## Real Example from download_met_images.py

### When You Run It:

```bash
python scripts/download_met_images.py --limit 1000 --download-missing
```

**Execution:**
1. Python reads file
2. Defines all functions
3. Reaches line 247: `if __name__ == "__main__":`
4. Checks: Is this being run directly? **YES**
5. Executes: `main()`
6. `main()` runs with your arguments: `--limit 1000 --download-missing`

### When You Import It:

```python
# In another script
from scripts.download_met_images import fetch_object, download_and_cache

# Use functions without running main()
obj = fetch_object(474)
path = download_and_cache("https://...")
```

**Execution:**
1. Python reads file
2. Defines all functions
3. Reaches line 247: `if __name__ == "__main__":`
4. Checks: Is this being run directly? **NO** (it's being imported)
5. Skips: `main()` is NOT called
6. Functions are available for you to use

---

## Visual Comparison

### Direct Execution:
```
Terminal: python script.py
           ↓
Python: Reads file → Defines functions → if __name__ == "__main__": → True → main()
```

### Import:
```
Python file: from script import function
              ↓
Python: Reads file → Defines functions → if __name__ == "__main__": → False → Skip
              ↓
Available: function() (but main() didn't run)
```

---

## Other Ways to Call main()

### Method 1: Direct Call (Not Recommended)

You could call `main()` directly in code:

```python
# At the end of the file
main()  # Always runs
```

**Problem:** Runs even when imported (bad!)

### Method 2: Conditional Call (Recommended - What We Use)

```python
# At the end of the file
if __name__ == "__main__":
    main()  # Only runs when executed directly
```

**Benefit:** Works as both script and module

### Method 3: Explicit Entry Point

```python
# At the end of the file
if __name__ == "__main__":
    import sys
    sys.exit(main())  # Exit with return code
```

**Benefit:** Can return exit codes for error handling

---

## Summary

| Question | Answer |
|----------|--------|
| **Where is main() called?** | At the bottom: `if __name__ == "__main__": main()` |
| **When does it run?** | Only when you run the script directly (not when imported) |
| **Why this pattern?** | Allows script to work both as executable AND as importable module |
| **Do I need to call main() manually?** | No! Python does it automatically when you run the script |

---

## Quick Reference

```python
# This is the standard Python pattern:

def main():
    # Your main code here
    pass

if __name__ == "__main__":
    main()  # ← This calls main() when script is run directly
```

**When you run:**
```bash
python script.py
```

**Python automatically:**
1. Reads the file
2. Defines `main()`
3. Sees `if __name__ == "__main__":`
4. Calls `main()`

**You don't need to do anything extra!** Just run the script and Python handles it.

