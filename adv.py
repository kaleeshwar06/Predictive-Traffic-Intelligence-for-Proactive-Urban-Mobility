import requests
from selenium import webdriver

# Use Selenium to scrape OpenCCTV (they don't have official API yet)
driver = webdriver.Chrome()
driver.get("https://opencctv.org/")

# Search for traffic cameras in your region
search_box = driver.find_element("name", "search")
search_box.send_keys("traffic camera Chennai")
search_box.submit()

# Extract stream URLs
streams = []
for iframe in driver.find_elements("tag name", "iframe"):
    src = iframe.get_attribute("src")
    if "stream" in src:
        streams.append(src)

print(f"Found {len(streams)} streams")