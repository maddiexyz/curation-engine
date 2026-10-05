from stores import STORES
import os
import requests
import json

OUTPUT_DIR = "sample-jsons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Starting fetch for sample product JSONs across {len(STORES)} stores into '{OUTPUT_DIR}/'...\n")

for store_key, store_info in STORES.items():
    name = store_info["name"]
    domain = store_info["domain"]
    platform = store_info.get("platform", "shopify")
    
    # Default URL based on prompt request
    url = f"https://{domain}/products.json?limit=1"
    filepath = os.path.join(OUTPUT_DIR, f"{store_key}.json")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            try:
                data = response.json()
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"[SUCCESS] {name} ({domain}) -> saved to {filepath}")
            except Exception:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"[SUCCESS] {name} ({domain}) -> saved raw response to {filepath}")
        else:
            # Fallback for WooCommerce if products.json is not available
            if platform == "woocommerce":
                wc_url = f"https://{domain}/wp-json/wc/store/v1/products?per_page=1"
                wc_resp = requests.get(wc_url, headers=headers, timeout=10)
                if wc_resp.status_code == 200:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(wc_resp.json(), f, indent=2, ensure_ascii=False)
                    print(f"[SUCCESS] {name} ({domain} WooCommerce) -> saved to {filepath}")
                else:
                    print(f"[FAILED]  {name} ({domain}) -> HTTP {response.status_code} (WC HTTP {wc_resp.status_code})")
            else:
                print(f"[FAILED]  {name} ({domain}) -> HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR]   {name} ({domain}) -> {e}")

print("\nDone!")
