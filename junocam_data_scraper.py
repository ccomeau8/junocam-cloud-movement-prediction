import random
import shutil
import requests

base_url = "https://www.missionjuno.swri.edu/Vault/VaultOutput?VaultID="

image_id = random.randint(41000, 42000)

req_url = base_url + str(image_id)

response = requests.get(req_url, stream=True)
# for image_id in range
with open(f"img.png", 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
