from stores import STORES
import os
import requests

OUTPUT_DIR = "robot-files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Starting fetch for {len(STORES)} stores into '{OUTPUT_DIR}/'...\n")

for store_key, store_info in STORES.items():
    name = store_info["name"]
    domain = store_info["domain"]
    url = f"https://{domain}/robots.txt"
    filepath = os.path.join(OUTPUT_DIR, f"{store_key}.txt")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"[SUCCESS] {name} ({domain}) -> saved to {filepath}")
        else:
            print(f"[FAILED]  {name} ({domain}) -> HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR]   {name} ({domain}) -> {e}")

print("\nDone!")
